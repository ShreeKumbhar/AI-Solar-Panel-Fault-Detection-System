from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.evaluation.metrics import calculate_regression_metrics
from src.evaluation.reports import build_comparison_report
from src.models.model_io import save_model
from src.models.model_registry import MODEL_VERSION, create_model, model_hyperparameters
from src.models.power_model import EXCLUDED_FEATURES, POWER_FEATURES, POWER_TARGET, prepare_power_datasets, validate_power_features


RANDOM_STATE = 42
DATASET_VERSION = "pv_fault_dataset-public-phase1"


def _load_xy(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path)
    validate_power_features(frame)
    return frame[POWER_FEATURES], frame[POWER_TARGET]


def main() -> None:
    processed_dir = project_root / "data" / "processed"
    power_dir = processed_dir / "power_model"
    artifact_dir = project_root / "artifacts"
    report_dir = project_root / "data" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    paths = prepare_power_datasets(processed_dir, power_dir)
    train_x, train_y = _load_xy(paths["train"])
    validation_x, validation_y = _load_xy(paths["validation"])

    model_results: dict[str, dict] = {}
    trained_models = {}
    for model_name in ["random_forest", "xgboost"]:
        model = create_model(model_name, random_state=RANDOM_STATE)
        started = time.perf_counter()
        model.fit(train_x, train_y)
        training_seconds = time.perf_counter() - started
        validation_predictions = model.predict(validation_x)
        validation_metrics = calculate_regression_metrics(validation_y, validation_predictions)
        model_results[model_name] = {
            "validation_metrics": validation_metrics,
            "training_seconds": float(training_seconds),
            "hyperparameters": model_hyperparameters(model),
        }
        trained_models[model_name] = model
        print(f"{model_name}: validation={validation_metrics}, training_seconds={training_seconds:.2f}")

    selected_model_name = min(
        model_results,
        key=lambda name: (model_results[name]["validation_metrics"]["rmse"], model_results[name]["validation_metrics"]["mae"]),
    )
    selection_criteria = "Select the model with the lowest validation RMSE; use validation MAE as the tie-breaker. Test metrics are not used."
    comparison = build_comparison_report(model_results, selected_model_name, selection_criteria)
    comparison["target"] = POWER_TARGET
    comparison["features"] = POWER_FEATURES
    comparison["healthy_selection_rule"] = "Rows with fault_label == 0 are retained in each chronological Phase 2 partition."
    comparison["normal_label"] = 0
    (report_dir / "power_model_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    development_x = pd.concat([train_x, validation_x], ignore_index=True)
    development_y = pd.concat([train_y, validation_y], ignore_index=True)
    final_model = create_model(selected_model_name, random_state=RANDOM_STATE)
    final_started = time.perf_counter()
    final_model.fit(development_x, development_y)
    final_training_seconds = time.perf_counter() - final_started
    model_path = save_model(final_model, artifact_dir / "expected_power_model.joblib")

    schema = {"target": POWER_TARGET, "features": POWER_FEATURES, "feature_order": POWER_FEATURES}
    (artifact_dir / "expected_power_feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    metadata = {
        "model_type": selected_model_name,
        "model_version": MODEL_VERSION,
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "feature_list": POWER_FEATURES,
        "target": POWER_TARGET,
        "normal_label": 0,
        "healthy_selection_rule": "fault_label == 0",
        "training_sample_count": int(len(train_x)),
        "validation_sample_count": int(len(validation_x)),
        "development_sample_count": int(len(development_x)),
        "test_sample_count": int(len(pd.read_csv(paths["test"]))),
        "validation_metrics": model_results[selected_model_name]["validation_metrics"],
        "final_test_metrics": None,
        "hyperparameters": model_hyperparameters(final_model),
        "random_state": RANDOM_STATE,
        "final_training_seconds": float(final_training_seconds),
        "dataset_version": DATASET_VERSION,
        "excluded_features": EXCLUDED_FEATURES,
        "retrained_on_train_plus_validation": True,
        "artifact": str(model_path.relative_to(project_root)),
    }
    (artifact_dir / "expected_power_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Selected model: {selected_model_name}")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
