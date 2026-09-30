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

from src.evaluation.fault_metrics import calculate_fault_metrics, class_distribution
from src.evaluation.fault_plots import plot_confusion_matrix
from src.evaluation.fault_reports import build_comparison_report
from src.models.fault_classifier import (
    CLASS_LABELS,
    CLASS_NAMES,
    CLASSIFIER_FEATURES,
    FEATURE_SCHEMA,
    MODEL_VERSION,
    FaultClassifier,
    apply_unknown_anomaly_rule,
    build_classification_features,
    calibrate_confidence_threshold,
    safe_hyperparameters,
)

RANDOM_STATE = 42
INPUT_DIR = project_root / "data" / "processed" / "anomaly_detection"
RAW_DIR = project_root / "data" / "processed"
OUTPUT_DIR = project_root / "data" / "processed" / "fault_classification"
ARTIFACT_DIR = project_root / "artifacts"
REPORT_DIR = project_root / "data" / "reports"


def _load_combined() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    parts = {}
    raw_columns = ["sample_index", "vdc1", "vdc2", "idc1", "idc2"]
    for split in ["train", "validation", "test"]:
        anomaly = pd.read_csv(INPUT_DIR / f"{split}_anomaly.csv")
        raw = pd.read_csv(RAW_DIR / f"{split}.csv", usecols=raw_columns)
        parts[split] = anomaly.merge(raw, on="sample_index", how="left", validate="one_to_one")
        if parts[split][["vdc1", "vdc2", "idc1", "idc2"]].isna().any().any():
            raise ValueError(f"Raw Phase 2 measurements could not be joined for {split}.")
    combined = pd.concat([parts[split] for split in ["train", "validation", "test"]], ignore_index=True)
    combined = combined.sort_values("sample_index").reset_index(drop=True)
    return parts, combined


def _prepare_partitions() -> dict[str, pd.DataFrame]:
    parts, combined = _load_combined()
    combined = build_classification_features(combined)
    rebuilt: dict[str, pd.DataFrame] = {}
    for split, original in parts.items():
        min_index = int(original["sample_index"].min())
        max_index = int(original["sample_index"].max())
        partition = combined[combined["sample_index"].between(min_index, max_index)].copy()
        if not partition["sample_index"].is_monotonic_increasing:
            raise ValueError(f"Partition lost chronology: {split}")
        rebuilt[split] = partition.reset_index(drop=True)
    return rebuilt


def _sample_weights(target: pd.Series) -> np.ndarray:
    counts = target.value_counts().to_dict()
    return target.map(lambda label: len(target) / (len(counts) * counts[int(label)])).to_numpy(dtype=float)


def _evaluate(classifier: FaultClassifier, frame: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray]:
    probabilities = classifier.predict_proba(frame[CLASSIFIER_FEATURES])
    predictions = classifier.predict(frame[CLASSIFIER_FEATURES])
    return calculate_fault_metrics(frame["fault_label"], predictions), predictions, probabilities


def _write_output(frame: pd.DataFrame, classifier: FaultClassifier, output_path: Path, threshold: float) -> None:
    probabilities = classifier.predict_proba(frame[CLASSIFIER_FEATURES])
    predictions = classifier.predict(frame[CLASSIFIER_FEATURES])
    anomaly = frame["anomaly"].astype(bool).to_numpy()
    decisions = apply_unknown_anomaly_rule(predictions, probabilities, anomaly, threshold)
    output = frame[["sample_index", *CLASSIFIER_FEATURES, "fault_label"]].copy()
    output["true_fault_label"] = output.pop("fault_label")
    output["true_fault_type"] = output["true_fault_label"].map(CLASS_LABELS)
    output["predicted_fault_label"] = decisions["predictions"]
    output["predicted_fault_type"] = output["predicted_fault_label"].map(CLASS_LABELS).fillna("UNKNOWN_ANOMALY")
    for index, name in enumerate(CLASS_NAMES):
        output[f"probability_{name.lower()}"] = probabilities[:, index]
    output["classification_confidence"] = decisions["confidence"]
    output["unknown_anomaly"] = decisions["unknown_anomaly"]
    output.to_csv(output_path, index=False)


def main() -> None:
    partitions = _prepare_partitions()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train = partitions["train"]
    validation = partitions["validation"]
    train_classes = sorted(train["fault_label"].unique().tolist())
    missing_train_classes = sorted(set(CLASS_LABELS) - set(train_classes))
    if missing_train_classes:
        print(f"Warning: chronological training split lacks labels {missing_train_classes}; no validation/test rows are moved into training.")

    results: dict[str, dict] = {}
    fitted: dict[str, FaultClassifier] = {}
    for model_name in ["random_forest", "xgboost"]:
        classifier = FaultClassifier(model_name=model_name, random_state=RANDOM_STATE)
        weights = _sample_weights(train["fault_label"]) if model_name == "xgboost" else None
        classifier.fit(train[CLASSIFIER_FEATURES], train["fault_label"], sample_weight=weights)
        metrics, predictions, probabilities = _evaluate(classifier, validation)
        results[model_name] = {"validation_metrics": metrics, "hyperparameters": safe_hyperparameters(classifier.model), "training_classes": train_classes}
        fitted[model_name] = classifier
        plot_confusion_matrix(metrics, REPORT_DIR / "plots" / "fault_classification", f"{model_name}_validation_confusion_matrix.png")
        plot_confusion_matrix(metrics, REPORT_DIR / "plots" / "fault_classification", f"{model_name}_validation_confusion_matrix_normalized.png", normalized=True)

    selected_model = max(results, key=lambda name: (results[name]["validation_metrics"]["f1_macro"], results[name]["validation_metrics"]["recall_macro"]))
    selected_validation = fitted[selected_model]
    validation_probabilities = selected_validation.predict_proba(validation[CLASSIFIER_FEATURES])
    validation_predictions = selected_validation.predict(validation[CLASSIFIER_FEATURES])
    confidence_threshold = calibrate_confidence_threshold(
        validation_probabilities,
        validation["fault_label"].to_numpy(),
        validation["anomaly"].astype(bool).to_numpy(),
    )
    comparison = build_comparison_report(
        results,
        selected_model,
        "Highest validation macro F1, with validation macro recall as tie-breaker; test data was not used.",
        confidence_threshold,
    )
    comparison["class_mapping"] = {str(label): name for label, name in CLASS_LABELS.items()}
    comparison["missing_training_classes"] = missing_train_classes
    (REPORT_DIR / "fault_classification_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    development = pd.concat([train, validation], ignore_index=True)
    final_classifier = FaultClassifier(model_name=selected_model, random_state=RANDOM_STATE)
    final_weights = _sample_weights(development["fault_label"]) if selected_model == "xgboost" else None
    final_classifier.fit(development[CLASSIFIER_FEATURES], development["fault_label"], sample_weight=final_weights)
    final_classifier.confidence_threshold = confidence_threshold
    final_classifier.save(ARTIFACT_DIR / "fault_classifier.joblib")

    feature_schema = {
        "target": "fault_label",
        "features": CLASSIFIER_FEATURES,
        "class_mapping": {str(label): name for label, name in CLASS_LABELS.items()},
        "feature_details": FEATURE_SCHEMA,
    }
    (ARTIFACT_DIR / "fault_classifier_feature_schema.json").write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")
    metadata = {
        "model_name": selected_model,
        "model_version": MODEL_VERSION,
        "algorithm": type(final_classifier.model).__name__,
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "class_mapping": {str(label): name for label, name in CLASS_LABELS.items()},
        "feature_list": CLASSIFIER_FEATURES,
        "hyperparameters": safe_hyperparameters(final_classifier.model),
        "validation_metrics": results[selected_model]["validation_metrics"],
        "confidence_threshold": confidence_threshold,
        "confidence_threshold_method": "Validation utility score over quantile candidates; anomaly rows require known-class correctness and low-confidence anomaly rows become UNKNOWN_ANOMALY.",
        "dataset_information": {"train_count": len(train), "validation_count": len(validation), "test_count": len(partitions["test"]), "train_classes": train_classes, "missing_training_classes": missing_train_classes},
        "retrained_on_train_plus_validation": True,
        "artifact": "artifacts/fault_classifier.joblib",
    }
    (ARTIFACT_DIR / "fault_classifier_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for split, frame in partitions.items():
        _write_output(frame, final_classifier, OUTPUT_DIR / f"{split}_fault.csv", confidence_threshold)

    print(json.dumps({"selected_model": selected_model, "confidence_threshold": confidence_threshold, "validation": results[selected_model]["validation_metrics"], "missing_training_classes": missing_train_classes}, indent=2))


if __name__ == "__main__":
    main()
