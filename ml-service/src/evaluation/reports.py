from __future__ import annotations

from datetime import date

import numpy as np


def build_comparison_report(model_results: dict, selected_model: str, selection_criteria: str) -> dict:
    return {
        "models_tested": model_results,
        "selected_model": selected_model,
        "selection_criteria": selection_criteria,
        "test_set_used_for_selection": False,
    }


def build_evaluation_report(
    model_name: str,
    test_metrics: dict,
    predictions,
    group_metrics: list[dict],
    feature_importance: list[dict],
) -> dict:
    residuals = np.asarray(predictions["residual_w"], dtype=float)
    absolute_errors = np.abs(residuals)
    return {
        "model": model_name,
        "sample_count": int(len(residuals)),
        "test_metrics": test_metrics,
        "residual_statistics": {
            "mean_w": float(residuals.mean()),
            "std_w": float(residuals.std()),
            "min_w": float(residuals.min()),
            "max_w": float(residuals.max()),
            "mae_w": float(absolute_errors.mean()),
        },
        "irradiance_group_metrics": group_metrics,
        "feature_importance": feature_importance,
        "generated_on": date.today().isoformat(),
    }
