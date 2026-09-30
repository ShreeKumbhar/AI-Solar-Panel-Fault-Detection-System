from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.evaluation.anomaly_metrics import calculate_anomaly_metrics
from src.evaluation.anomaly_reports import build_anomaly_comparison_report
from src.models.anomaly_detector import (
    ANOMALY_FEATURES,
    MODEL_VERSION,
    IsolationForestAnomalyDetector,
    calibrate_threshold,
    generate_anomaly_frame,
)
from src.models.model_io import load_model
from src.models.power_model import POWER_FEATURES

RANDOM_STATE = 42
ROLLING_WINDOW = 10
FAULT_LABELS = {
    0: "Normal",
    1: "Short-Circuit",
    2: "Degradation",
    3: "Open-Circuit",
    4: "Shadowing",
}


def _load_partition(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not frame["sample_index"].is_monotonic_increasing:
        raise ValueError(f"Partition is not chronological: {path}")
    return frame


def _output_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        "sample_index",
        "actual_power_w",
        "expected_power_w",
        "irradiance",
        "pv_temperature",
        "residual_w",
        "absolute_error_w",
        "relative_deviation",
        "deviation_percent",
        "rolling_mean_residual",
        "rolling_std_residual",
        "rolling_mean_absolute_error",
        "rolling_mean_deviation",
        "previous_residual",
        "previous_deviation",
        "deviation_trend",
        *ANOMALY_FEATURES,
        "raw_isolation_score",
        "anomaly_score",
        "anomaly",
    ]
    if "fault_label" in frame.columns:
        columns.extend(["fault_label", "is_anomaly_true"])
    return list(dict.fromkeys(column for column in columns if column in frame.columns))


def _with_predictions(frame: pd.DataFrame, detector: IsolationForestAnomalyDetector, baseline_threshold: float) -> pd.DataFrame:
    output = frame.copy()
    output["raw_isolation_score"] = detector.score_samples(output[ANOMALY_FEATURES])
    output["anomaly_score"] = detector.calculate_anomaly_score(output[ANOMALY_FEATURES])
    output["anomaly"] = detector.predict_anomaly(output[ANOMALY_FEATURES]).astype(bool)
    output["baseline_anomaly"] = (output["deviation_percent"].abs() >= baseline_threshold).astype(bool)
    return output


def main() -> None:
    processed_dir = project_root / "data" / "processed"
    output_dir = processed_dir / "anomaly_detection"
    report_dir = project_root / "data" / "reports"
    artifact_dir = project_root / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    expected_model = load_model(artifact_dir / "expected_power_model.joblib")
    partitions = {
        split: _load_partition(processed_dir / f"{split}.csv")
        for split in ["train", "validation", "test"]
    }
    anomaly_frames = {
        split: generate_anomaly_frame(partition, expected_model, POWER_FEATURES, rolling_window=ROLLING_WINDOW)
        for split, partition in partitions.items()
    }

    detector = IsolationForestAnomalyDetector(
        feature_names=ANOMALY_FEATURES,
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    detector.fit(anomaly_frames["train"][ANOMALY_FEATURES])
    validation_labels = anomaly_frames["validation"]["is_anomaly_true"].to_numpy()
    detector_threshold = detector.calibrate(anomaly_frames["validation"][ANOMALY_FEATURES], validation_labels)
    baseline_scores = anomaly_frames["validation"]["deviation_percent"].abs().to_numpy()
    baseline_threshold = calibrate_threshold(baseline_scores, validation_labels)

    detector_validation_scores = detector.calculate_anomaly_score(anomaly_frames["validation"][ANOMALY_FEATURES])
    detector_validation_metrics = calculate_anomaly_metrics(validation_labels, detector_validation_scores >= detector_threshold)
    baseline_validation_metrics = calculate_anomaly_metrics(validation_labels, baseline_scores >= baseline_threshold)
    selected_detector = "isolation_forest" if (detector_validation_metrics["f1"], detector_validation_metrics["recall"]) >= (baseline_validation_metrics["f1"], baseline_validation_metrics["recall"]) else "residual_threshold_baseline"
    comparison = build_anomaly_comparison_report(
        {"validation_metrics": baseline_validation_metrics, "threshold": baseline_threshold, "score": "abs(deviation_percent)"},
        {"validation_metrics": detector_validation_metrics, "threshold": detector_threshold, "score": "normalized(-IsolationForest.score_samples())"},
        selected_detector,
        detector_threshold if selected_detector == "isolation_forest" else baseline_threshold,
        "Validation F1 maximized, with validation recall as the tie-breaker; the test set was not used.",
    )
    comparison["feature_list"] = ANOMALY_FEATURES
    comparison["fault_label_mapping"] = FAULT_LABELS
    (report_dir / "anomaly_detection_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    detector.save(artifact_dir / "anomaly_detector.joblib")
    metadata = {
        "model_name": "IsolationForest",
        "model_version": MODEL_VERSION,
        "algorithm": "sklearn.ensemble.IsolationForest",
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "n_estimators": 200,
        "features": ANOMALY_FEATURES,
        "power_model_features": POWER_FEATURES,
        "threshold": detector_threshold,
        "baseline_threshold": baseline_threshold,
        "selected_detector": selected_detector,
        "threshold_calibration_method": "Validation F1 maximization with validation recall tie-breaker.",
        "raw_score_definition": "IsolationForest.score_samples; higher values indicate more normal observations.",
        "normalized_score_definition": "Min-max normalized negative raw Isolation Forest score using training-score bounds; higher means more anomalous.",
        "rolling_window": ROLLING_WINDOW,
        "training_split": "70% chronological Phase 2 train partition",
        "validation_split": "15% chronological Phase 2 validation partition",
        "test_split": "15% chronological Phase 2 test partition",
        "partition_counts": {split: int(len(frame)) for split, frame in anomaly_frames.items()},
        "fault_labels_used_only_for": ["validation threshold calibration", "evaluation"],
        "dataset_version": "pv_fault_dataset-public-phase1",
        "fault_label_mapping": FAULT_LABELS,
        "artifact": "artifacts/anomaly_detector.joblib",
    }
    (artifact_dir / "anomaly_detector_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for split, frame in anomaly_frames.items():
        predicted = _with_predictions(frame, detector, baseline_threshold)
        predicted[_output_columns(predicted)].to_csv(output_dir / f"{split}_anomaly.csv", index=False)

    print(json.dumps({"selected_detector": selected_detector, "threshold": comparison["selected_threshold"], "baseline_validation": baseline_validation_metrics, "isolation_forest_validation": detector_validation_metrics}, indent=2))


if __name__ == "__main__":
    main()
