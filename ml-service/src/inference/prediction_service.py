from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.anomaly_detector import ANOMALY_FEATURES, IsolationForestAnomalyDetector, build_anomaly_features
from src.models.fault_classifier import CLASSIFIER_FEATURES, CLASS_LABELS, FaultClassifier, apply_unknown_anomaly_rule
from src.models.model_io import load_model
from src.models.power_model import POWER_FEATURES

LOGGER = logging.getLogger(__name__)
HISTORY_WINDOW = 10
MODEL_VERSION = "phase7-inference-v1"


@dataclass(frozen=True)
class ModelBundle:
    expected_power_model: object
    expected_power_metadata: dict
    expected_power_schema: dict
    anomaly_detector: IsolationForestAnomalyDetector
    anomaly_metadata: dict
    fault_classifier: FaultClassifier
    fault_metadata: dict
    fault_schema: dict

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "ModelBundle":
        root = Path(artifact_dir)
        required = [
            "expected_power_model.joblib",
            "expected_power_metadata.json",
            "expected_power_feature_schema.json",
            "anomaly_detector.joblib",
            "anomaly_detector_metadata.json",
            "fault_classifier.joblib",
            "fault_classifier_metadata.json",
            "fault_classifier_feature_schema.json",
        ]
        missing = [name for name in required if not (root / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing model artifacts: {', '.join(missing)}")
        expected_schema = json.loads((root / "expected_power_feature_schema.json").read_text(encoding="utf-8"))
        fault_schema = json.loads((root / "fault_classifier_feature_schema.json").read_text(encoding="utf-8"))
        if expected_schema.get("features") != POWER_FEATURES:
            raise ValueError("Expected-power artifact feature schema is incompatible with the frozen pipeline.")
        if fault_schema.get("features") != CLASSIFIER_FEATURES:
            raise ValueError("Fault-classifier artifact feature schema is incompatible with the frozen pipeline.")
        anomaly_detector = IsolationForestAnomalyDetector.load(root / "anomaly_detector.joblib")
        if anomaly_detector.feature_names != ANOMALY_FEATURES:
            raise ValueError("Anomaly detector feature schema is incompatible with the frozen pipeline.")
        return cls(
            expected_power_model=load_model(root / "expected_power_model.joblib"),
            expected_power_metadata=json.loads((root / "expected_power_metadata.json").read_text(encoding="utf-8")),
            expected_power_schema=expected_schema,
            anomaly_detector=anomaly_detector,
            anomaly_metadata=json.loads((root / "anomaly_detector_metadata.json").read_text(encoding="utf-8")),
            fault_classifier=FaultClassifier.load(root / "fault_classifier.joblib"),
            fault_metadata=json.loads((root / "fault_classifier_metadata.json").read_text(encoding="utf-8")),
            fault_schema=fault_schema,
        )


class PredictionService:
    def __init__(self, bundle: ModelBundle, history_window: int = HISTORY_WINDOW) -> None:
        self.bundle = bundle
        self.history_window = history_window
        self._histories: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=history_window - 1))
        self._last_timestamps: dict[str, object] = {}
        self._lock = threading.RLock()

    def clear_history(self, panel_id: str | None = None) -> None:
        with self._lock:
            if panel_id is None:
                self._histories.clear()
                self._last_timestamps.clear()
            else:
                self._histories.pop(panel_id, None)
                self._last_timestamps.pop(panel_id, None)

    def model_info(self) -> dict:
        return {
            "expected_power_model": self.bundle.expected_power_metadata.get("model_type", type(self.bundle.expected_power_model).__name__),
            "anomaly_model": self.bundle.anomaly_metadata.get("model_name", "IsolationForest"),
            "fault_classifier": self.bundle.fault_metadata.get("model_name", type(self.bundle.fault_classifier.model).__name__),
            "fault_classes": list(CLASS_LABELS.values()),
            "fault_confidence_threshold": float(self.bundle.fault_metadata["confidence_threshold"]),
            "model_versions": {
                "expected_power": self.bundle.expected_power_metadata.get("model_version", "unknown"),
                "anomaly": self.bundle.anomaly_metadata.get("model_version", "unknown"),
                "fault_classifier": self.bundle.fault_metadata.get("model_version", "unknown"),
                "service": MODEL_VERSION,
            },
        }

    @staticmethod
    def _actual_power(observation) -> float:
        return float(observation.vdc1 * observation.idc1 + observation.vdc2 * observation.idc2)

    def _build_current_frame(self, observation, expected_power: float, actual_power: float) -> pd.DataFrame:
        history = list(self._histories[observation.panel_id])
        current = {
            "sample_index": len(history),
            "vdc1": observation.vdc1,
            "vdc2": observation.vdc2,
            "idc1": observation.idc1,
            "idc2": observation.idc2,
            "actual_power_w": actual_power,
            "expected_power_w": expected_power,
            "irradiance": observation.irradiance,
            "pv_temperature": observation.pv_temperature,
        }
        frame = pd.DataFrame([*history, current])
        return build_anomaly_features(frame, rolling_window=self.history_window).tail(1).reset_index(drop=True)

    @staticmethod
    def _severity(anomaly: bool, deviation_percent: float, fault_type: str) -> str:
        if not anomaly:
            return "NORMAL"
        if fault_type in {"SHORT_CIRCUIT", "OPEN_CIRCUIT"} or abs(deviation_percent) >= 50.0:
            return "CRITICAL"
        return "WARNING"

    def predict(self, observation) -> dict:
        with self._lock:
            previous_timestamp = self._last_timestamps.get(observation.panel_id)
            if previous_timestamp is not None and observation.timestamp <= previous_timestamp:
                raise ValueError("observations for each panel must have strictly increasing timestamps")
            raw = pd.DataFrame([
                {
                    "vdc1": observation.vdc1,
                    "vdc2": observation.vdc2,
                    "idc1": observation.idc1,
                    "idc2": observation.idc2,
                    "irradiance": observation.irradiance,
                    "pv_temperature": observation.pv_temperature,
                }
            ])
            actual_power = self._actual_power(observation)
            expected_power = float(self.bundle.expected_power_model.predict(raw[POWER_FEATURES])[0])
            anomaly_frame = self._build_current_frame(observation, expected_power, actual_power)
            anomaly_score = float(self.bundle.anomaly_detector.calculate_anomaly_score(anomaly_frame[ANOMALY_FEATURES])[0])
            anomaly = bool(self.bundle.anomaly_detector.predict_anomaly(anomaly_frame[ANOMALY_FEATURES])[0])
            classifier_frame = anomaly_frame.copy()
            classifier_frame["anomaly_score"] = anomaly_score
            classifier_frame["anomaly"] = anomaly
            probabilities = self.bundle.fault_classifier.predict_proba(classifier_frame[CLASSIFIER_FEATURES])
            predicted = self.bundle.fault_classifier.predict(classifier_frame[CLASSIFIER_FEATURES])
            decision = apply_unknown_anomaly_rule(
                predicted,
                probabilities,
                np.asarray([anomaly]),
                float(self.bundle.fault_metadata["confidence_threshold"]),
            )
            final_label = int(decision["predictions"][0])
            fault_type = "UNKNOWN_ANOMALY" if final_label == -1 else (CLASS_LABELS[final_label] if anomaly else "NORMAL")
            residual = actual_power - expected_power
            deviation_percent = float(residual / expected_power * 100.0) if expected_power > 1e-9 else 0.0
            result = {
                "panel_id": observation.panel_id,
                "timestamp": observation.timestamp,
                "actual_power_w": actual_power,
                "expected_power_w": expected_power,
                "deviation_w": residual,
                "deviation_percent": deviation_percent,
                "anomaly": anomaly,
                "anomaly_score": anomaly_score,
                "fault_type": fault_type,
                "confidence": float(decision["confidence"][0]),
                "severity": self._severity(anomaly, deviation_percent, fault_type),
                "model_version": MODEL_VERSION,
            }
            self._histories[observation.panel_id].append({
                "sample_index": len(self._histories[observation.panel_id]),
                "actual_power_w": actual_power,
                "expected_power_w": expected_power,
                "irradiance": observation.irradiance,
                "pv_temperature": observation.pv_temperature,
            })
            self._last_timestamps[observation.panel_id] = observation.timestamp
            return result

    def predict_batch(self, observations) -> list[dict]:
        return [self.predict(observation) for observation in observations]
