from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.evaluation.anomaly_metrics import calculate_anomaly_metrics
from src.evaluation.anomaly_plots import create_anomaly_plots, create_confusion_matrix_plot
from src.evaluation.anomaly_reports import build_anomaly_evaluation_report
from src.models.anomaly_detector import ANOMALY_FEATURES, IsolationForestAnomalyDetector

FAULT_LABELS = {0: "Normal", 1: "Short-Circuit", 2: "Degradation", 3: "Open-Circuit", 4: "Shadowing"}


def _group_metrics(frame: pd.DataFrame) -> list[dict]:
    labels = pd.qcut(frame["irradiance"], q=3, labels=["low", "medium", "high"], duplicates="drop")
    result = []
    for group_name, group in frame.assign(_irradiance_group=labels).groupby("_irradiance_group", observed=False):
        metrics = calculate_anomaly_metrics(group["is_anomaly_true"], group["anomaly"])
        result.append({"group": str(group_name), **metrics})
    return result


def _fault_breakdown(frame: pd.DataFrame) -> list[dict]:
    result = []
    for label, name in FAULT_LABELS.items():
        actual = (frame["fault_label"] == label).astype(int)
        metrics = calculate_anomaly_metrics(actual, frame["anomaly"])
        result.append({"fault_label": label, "fault_type": name, **metrics, "fault_sample_count": int(actual.sum()), "anomaly_recall": float(frame.loc[actual == 1, "anomaly"].mean()) if actual.sum() else 0.0})
    return result


def main() -> None:
    artifact_dir = project_root / "artifacts"
    report_dir = project_root / "data" / "reports"
    plot_dir = report_dir / "plots" / "anomaly_detection"
    test_path = project_root / "data" / "processed" / "anomaly_detection" / "test_anomaly.csv"
    frame = pd.read_csv(test_path)
    detector = IsolationForestAnomalyDetector.load(artifact_dir / "anomaly_detector.joblib")
    frame["raw_isolation_score"] = detector.score_samples(frame[ANOMALY_FEATURES])
    frame["anomaly_score"] = detector.calculate_anomaly_score(frame[ANOMALY_FEATURES])
    frame["anomaly"] = detector.predict_anomaly(frame[ANOMALY_FEATURES]).astype(bool)
    frame["is_anomaly_true"] = (frame["fault_label"] != 0).astype(int)

    final_metrics = calculate_anomaly_metrics(frame["is_anomaly_true"], frame["anomaly"])
    fault_breakdown = _fault_breakdown(frame)
    irradiance_breakdown = _group_metrics(frame)
    false_positives = frame[(frame["is_anomaly_true"] == 0) & frame["anomaly"]]
    false_negatives = frame[(frame["is_anomaly_true"] == 1) & ~frame["anomaly"]]
    error_analysis = {
        "false_positive_count": int(len(false_positives)),
        "false_negative_count": int(len(false_negatives)),
        "false_positive_irradiance_mean": float(false_positives["irradiance"].mean()) if len(false_positives) else None,
        "false_negative_irradiance_mean": float(false_negatives["irradiance"].mean()) if len(false_negatives) else None,
        "largest_false_positive_errors": false_positives.nlargest(10, "absolute_error_w")[
            ["sample_index", "actual_power_w", "expected_power_w", "deviation_percent", "irradiance", "anomaly_score"]
        ].to_dict(orient="records"),
        "largest_false_negative_errors": false_negatives.nlargest(10, "absolute_error_w")[
            ["sample_index", "actual_power_w", "expected_power_w", "deviation_percent", "irradiance", "anomaly_score", "fault_label"]
        ].to_dict(orient="records"),
    }
    metadata = json.loads((artifact_dir / "anomaly_detector_metadata.json").read_text(encoding="utf-8"))
    evaluation = build_anomaly_evaluation_report(final_metrics, fault_breakdown, irradiance_breakdown, error_analysis, metadata)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "anomaly_detection_evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    create_anomaly_plots(frame, plot_dir)
    create_confusion_matrix_plot(final_metrics, plot_dir)
    frame.to_csv(report_dir / "anomaly_detection_test_predictions.csv", index=False)
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
