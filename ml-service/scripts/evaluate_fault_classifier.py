from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.evaluation.fault_metrics import calculate_fault_metrics, class_distribution
from src.evaluation.fault_plots import plot_class_distribution, plot_confusion_matrix, plot_feature_importance
from src.evaluation.fault_reports import build_evaluation_report
from src.evaluation.fault_summary import write_fault_summary
from src.models.fault_classifier import CLASS_LABELS, CLASS_NAMES, CLASSIFIER_FEATURES, FaultClassifier, apply_unknown_anomaly_rule

OUTPUT_DIR = project_root / "data" / "processed" / "fault_classification"
REPORT_DIR = project_root / "data" / "reports"
PLOT_DIR = REPORT_DIR / "plots" / "fault_classification"


def _irradiance_breakdown(frame: pd.DataFrame) -> list[dict]:
    groups = pd.qcut(frame["irradiance"], q=3, labels=["low", "medium", "high"], duplicates="drop")
    result = []
    for group_name, group in frame.assign(_irradiance_group=groups).groupby("_irradiance_group", observed=False):
        metrics = calculate_fault_metrics(group["true_fault_label"], group["predicted_fault_label"])
        result.append({"group": str(group_name), "sample_count": len(group), **metrics})
    return result


def _error_analysis(frame: pd.DataFrame) -> dict:
    errors = frame[frame["true_fault_label"] != frame["predicted_fault_label"]]
    confusions = (
        errors.groupby(["true_fault_type", "predicted_fault_type"], dropna=False).size().sort_values(ascending=False).reset_index(name="count").to_dict(orient="records")
    )
    return {"incorrect_count": len(errors), "confusions": confusions[:20], "largest_errors": errors.nlargest(20, "absolute_error_w")[
        ["sample_index", "true_fault_type", "predicted_fault_type", "irradiance", "pv_temperature", "actual_power_w", "expected_power_w", "residual_w"]
    ].to_dict(orient="records")}


def main() -> None:
    test = pd.read_csv(OUTPUT_DIR / "test_fault.csv")
    classifier = FaultClassifier.load(project_root / "artifacts" / "fault_classifier.joblib")
    probabilities = classifier.predict_proba(test[CLASSIFIER_FEATURES])
    predictions = classifier.predict(test[CLASSIFIER_FEATURES])
    decisions = apply_unknown_anomaly_rule(predictions, probabilities, test["anomaly"].astype(bool), classifier.confidence_threshold)

    result = test.copy()
    result["predicted_fault_label"] = decisions["predictions"]
    result["predicted_fault_type"] = result["predicted_fault_label"].map(CLASS_LABELS).fillna("UNKNOWN_ANOMALY")
    result["classification_confidence"] = decisions["confidence"]
    result["unknown_anomaly"] = decisions["unknown_anomaly"]
    metrics = calculate_fault_metrics(result["true_fault_label"], predictions)
    distributions = {split: class_distribution(pd.read_csv(OUTPUT_DIR / f"{split}_fault.csv")["true_fault_label"]) for split in ["train", "validation", "test"]}
    importance = [{"feature": feature, "importance": float(value)} for feature, value in sorted(zip(CLASSIFIER_FEATURES, classifier.model.feature_importances_), key=lambda item: item[1], reverse=True)]
    evaluation = build_evaluation_report(
        metrics,
        distributions,
        _irradiance_breakdown(result),
        _error_analysis(result),
        importance,
        {"count": int(result["unknown_anomaly"].sum()), "rate": float(result["unknown_anomaly"].mean())},
        json.loads((project_root / "artifacts" / "fault_classifier_metadata.json").read_text(encoding="utf-8")),
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "fault_classification_evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    (REPORT_DIR / "fault_feature_importance.json").write_text(json.dumps(importance, indent=2), encoding="utf-8")
    comparison = json.loads((REPORT_DIR / "fault_classification_comparison.json").read_text(encoding="utf-8"))
    write_fault_summary(REPORT_DIR / "fault_classification_summary.md", comparison, evaluation)
    plot_class_distribution(distributions, PLOT_DIR)
    plot_confusion_matrix(metrics, PLOT_DIR, "fault_confusion_matrix_test.png")
    plot_confusion_matrix(metrics, PLOT_DIR, "fault_confusion_matrix_test_normalized.png", normalized=True)
    plot_feature_importance(importance, PLOT_DIR)
    result.to_csv(REPORT_DIR / "fault_classification_test_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(json.dumps({"unknown_anomaly_count": int(result["unknown_anomaly"].sum()), "unknown_anomaly_rate": float(result["unknown_anomaly"].mean())}, indent=2))


if __name__ == "__main__":
    main()
