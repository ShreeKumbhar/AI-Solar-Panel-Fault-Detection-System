from __future__ import annotations

from datetime import date


def build_comparison_report(results: dict, selected_model: str, selection_criterion: str, threshold: float) -> dict:
    return {
        "models_tested": results,
        "selected_model": selected_model,
        "selection_criterion": selection_criterion,
        "selected_confidence_threshold": threshold,
        "test_set_used_for_selection": False,
    }


def build_evaluation_report(
    final_metrics: dict,
    distributions: dict,
    irradiance_breakdown: list[dict],
    error_analysis: dict,
    feature_importance: list[dict],
    unknown_statistics: dict,
    metadata: dict,
) -> dict:
    return {
        "evaluation_scope": "FINAL UNTOUCHED TEST RESULT",
        "final_test_metrics": final_metrics,
        "class_distribution": distributions,
        "irradiance_breakdown": irradiance_breakdown,
        "error_analysis": error_analysis,
        "feature_importance": feature_importance,
        "unknown_anomaly_statistics": unknown_statistics,
        "model_metadata": metadata,
        "generated_on": date.today().isoformat(),
    }
