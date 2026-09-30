from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.evaluation.metrics import calculate_group_metrics, calculate_regression_metrics
from src.evaluation.plots import create_feature_importance_plot, create_power_plots
from src.evaluation.reports import build_evaluation_report
from src.models.model_io import load_model
from src.models.power_model import POWER_FEATURES, POWER_TARGET, validate_power_features


def main() -> None:
    artifact_dir = project_root / "artifacts"
    report_dir = project_root / "data" / "reports"
    test_path = project_root / "data" / "processed" / "power_model" / "test_power.csv"
    test_frame = pd.read_csv(test_path)
    validate_power_features(test_frame)

    metadata_path = artifact_dir / "expected_power_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = load_model(artifact_dir / "expected_power_model.joblib")
    predictions = model.predict(test_frame[POWER_FEATURES])

    result = test_frame[["sample_index", *POWER_FEATURES, POWER_TARGET]].copy()
    result["expected_power_w"] = predictions
    result["residual_w"] = result[POWER_TARGET] - result["expected_power_w"]
    result["absolute_error_w"] = result["residual_w"].abs()
    result["relative_error_percent"] = np.where(
        result[POWER_TARGET].abs() > 1e-9,
        result["absolute_error_w"] / result[POWER_TARGET].abs() * 100.0,
        np.nan,
    )

    test_metrics = calculate_regression_metrics(result[POWER_TARGET], result["expected_power_w"])
    group_metrics = calculate_group_metrics(result.rename(columns={POWER_TARGET: "actual_power_w"}))
    importance = [
        {"feature": feature, "importance": float(value)}
        for feature, value in sorted(zip(POWER_FEATURES, model.feature_importances_), key=lambda item: item[1], reverse=True)
    ]
    evaluation = build_evaluation_report(
        metadata["model_type"],
        test_metrics,
        result,
        group_metrics,
        importance,
    )
    evaluation["largest_absolute_errors"] = (
        result.nlargest(10, "absolute_error_w")[["sample_index", POWER_TARGET, "expected_power_w", "residual_w", "irradiance", "pv_temperature"]]
        .to_dict(orient="records")
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(report_dir / "power_model_test_predictions.csv", index=False)
    (report_dir / "power_model_evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    create_power_plots(result.rename(columns={POWER_TARGET: "actual_power_w"}), report_dir)
    create_feature_importance_plot(importance, report_dir)

    metadata["final_test_metrics"] = test_metrics
    metadata["test_evaluation_report"] = "data/reports/power_model_evaluation.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
