from __future__ import annotations

from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

CLASS_LABELS = {
    0: "NORMAL",
    1: "SHORT_CIRCUIT",
    2: "DEGRADATION",
    3: "OPEN_CIRCUIT",
    4: "SHADING",
}
CLASS_NAMES = list(CLASS_LABELS.values())
CLASSIFIER_FEATURES = [
    "vdc1",
    "vdc2",
    "idc1",
    "idc2",
    "irradiance",
    "pv_temperature",
    "actual_power_w",
    "expected_power_w",
    "residual_w",
    "absolute_error_w",
    "deviation_percent",
    "rolling_mean_residual",
    "rolling_std_residual",
    "rolling_mean_absolute_error",
    "rolling_mean_deviation",
    "previous_residual",
    "previous_deviation",
    "deviation_trend",
    "anomaly_score",
    "anomaly",
]
TARGET_COLUMN = "fault_label"
MODEL_VERSION = "phase5-fault-classifier-v1"

FEATURE_SCHEMA = {
    "vdc1": {"dtype": "float64", "source": "Phase 1 electrical measurement", "description": "String 1 DC voltage", "inference_available": True},
    "vdc2": {"dtype": "float64", "source": "Phase 1 electrical measurement", "description": "String 2 DC voltage", "inference_available": True},
    "idc1": {"dtype": "float64", "source": "Phase 1 electrical measurement", "description": "String 1 DC current", "inference_available": True},
    "idc2": {"dtype": "float64", "source": "Phase 1 electrical measurement", "description": "String 2 DC current", "inference_available": True},
    "irradiance": {"dtype": "float64", "source": "Phase 1 ambient measurement", "description": "Solar irradiance", "inference_available": True},
    "pv_temperature": {"dtype": "float64", "source": "Phase 1 ambient measurement", "description": "PV module temperature", "inference_available": True},
    "actual_power_w": {"dtype": "float64", "source": "Phase 1 calculated measurement", "description": "Measured string power total", "inference_available": True},
    "expected_power_w": {"dtype": "float64", "source": "Phase 3 expected-power model", "description": "Expected healthy power", "inference_available": True},
    "residual_w": {"dtype": "float64", "source": "Phase 4 residual calculation", "description": "Actual minus expected power", "inference_available": True},
    "absolute_error_w": {"dtype": "float64", "source": "Phase 4 residual calculation", "description": "Absolute power residual", "inference_available": True},
    "deviation_percent": {"dtype": "float64", "source": "Phase 4 residual calculation", "description": "Power deviation percentage", "inference_available": True},
    "rolling_mean_residual": {"dtype": "float64", "source": "Phase 4 causal history", "description": "Trailing residual mean", "inference_available": True},
    "rolling_std_residual": {"dtype": "float64", "source": "Phase 4 causal history", "description": "Trailing residual standard deviation", "inference_available": True},
    "rolling_mean_absolute_error": {"dtype": "float64", "source": "Phase 4 causal history", "description": "Trailing absolute-error mean", "inference_available": True},
    "rolling_mean_deviation": {"dtype": "float64", "source": "Phase 4 causal history", "description": "Trailing deviation mean", "inference_available": True},
    "previous_residual": {"dtype": "float64", "source": "Phase 4 causal history", "description": "Previous residual", "inference_available": True},
    "previous_deviation": {"dtype": "float64", "source": "Phase 4 causal history", "description": "Previous deviation", "inference_available": True},
    "deviation_trend": {"dtype": "float64", "source": "Phase 4 causal history", "description": "Trailing deviation trend", "inference_available": True},
    "anomaly_score": {"dtype": "float64", "source": "Phase 4 anomaly detector", "description": "Normalized anomaly score", "inference_available": True},
    "anomaly": {"dtype": "bool", "source": "Phase 4 anomaly detector", "description": "Binary anomaly decision", "inference_available": True},
}


def build_classification_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in [*CLASSIFIER_FEATURES, TARGET_COLUMN] if column not in frame.columns]
    if missing:
        raise ValueError(f"Classification input is missing columns: {missing}")
    if "sample_index" in frame.columns and not frame["sample_index"].is_monotonic_increasing:
        raise ValueError("Classification data must preserve chronological sample_index order.")
    result = frame.copy()
    numeric = result[CLASSIFIER_FEATURES].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Classification features contain NaN or infinite values.")
    return result


def build_classifier(model_name: str, random_state: int = 42):
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=24,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=5,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=random_state,
            tree_method="hist",
        )
    raise ValueError(f"Unknown classifier: {model_name}")


def safe_hyperparameters(model) -> dict:
    parameters = {}
    for key, value in model.get_params().items():
        if isinstance(value, float) and not np.isfinite(value):
            parameters[key] = None
        elif isinstance(value, (str, int, float, bool, type(None))):
            parameters[key] = value
    return parameters


def classification_metrics(y_true, predictions) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_macro": float(precision_score(y_true, predictions, labels=list(CLASS_LABELS), average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, predictions, labels=list(CLASS_LABELS), average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, predictions, labels=list(CLASS_LABELS), average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, predictions, labels=list(CLASS_LABELS), average="weighted", zero_division=0)),
        "classification_report": classification_report(y_true, predictions, labels=list(CLASS_LABELS), target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=list(CLASS_LABELS)).tolist(),
    }


def calibrate_confidence_threshold(probabilities: np.ndarray, y_true, anomaly: np.ndarray) -> float:
    confidence = np.asarray(probabilities).max(axis=1)
    labels = np.asarray(y_true)
    anomaly = np.asarray(anomaly, dtype=bool)
    candidates = np.unique(np.round(np.quantile(confidence, np.linspace(0.05, 0.95, 37)), 6))
    best = (float("-inf"), 0.5)
    for threshold in candidates:
        unknown = anomaly & (confidence < threshold)
        predicted = np.argmax(probabilities, axis=1).astype(int)
        known_correct = ((~anomaly) & (predicted == 0)) | (anomaly & ~unknown & (predicted == labels))
        score = float(known_correct.mean()) - float(unknown.mean()) * 0.1
        if score > best[0]:
            best = (score, float(threshold))
    return best[1]


def apply_unknown_anomaly_rule(predictions, probabilities: np.ndarray, anomaly, threshold: float) -> dict[str, np.ndarray]:
    predictions = np.asarray(predictions, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    anomaly = np.asarray(anomaly, dtype=bool)
    confidence = probabilities.max(axis=1)
    unknown = anomaly & (confidence < threshold)
    final_predictions = predictions.copy()
    final_predictions[unknown] = -1
    return {"predictions": final_predictions, "confidence": confidence, "unknown_anomaly": unknown}


class FaultClassifier:
    def __init__(self, model_name: str = "xgboost", random_state: int = 42) -> None:
        self.model_name = model_name
        self.random_state = random_state
        self.model = build_classifier(model_name, random_state=random_state)
        self.feature_names = list(CLASSIFIER_FEATURES)
        self.classes_ = list(CLASS_LABELS)
        self.confidence_threshold: float | None = None
        self.internal_to_label: dict[int, int] = {}

    def fit(self, features: pd.DataFrame, target, sample_weight=None) -> "FaultClassifier":
        matrix = build_classification_features(features.assign(fault_label=target))[self.feature_names]
        labels = sorted(np.unique(np.asarray(target, dtype=int)).tolist())
        self.internal_to_label = {index: label for index, label in enumerate(labels)}
        encoded_target = np.asarray([labels.index(int(label)) for label in target], dtype=int)
        self.model.fit(matrix, encoded_target, sample_weight=sample_weight)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        matrix = features[self.feature_names]
        probabilities = np.asarray(self.model.predict_proba(matrix), dtype=float)
        aligned = np.zeros((len(matrix), len(CLASS_LABELS)), dtype=float)
        for index, internal_label in enumerate(self.model.classes_):
            aligned[:, self.internal_to_label[int(internal_label)]] = probabilities[:, index]
        return aligned

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        internal_predictions = np.asarray(self.model.predict(features[self.feature_names]), dtype=int)
        return np.asarray([self.internal_to_label[int(label)] for label in internal_predictions], dtype=int)

    @staticmethod
    def confidence(probabilities: np.ndarray) -> np.ndarray:
        return np.asarray(probabilities).max(axis=1)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "FaultClassifier":
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Fault classifier artifact not found: {source}")
        return joblib.load(source)
