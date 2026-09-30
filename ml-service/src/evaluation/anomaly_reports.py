from __future__ import annotations

from datetime import date


def build_anomaly_comparison_report(baseline: dict, isolation_forest: dict, selected_detector: str, threshold: float, method: str) -> dict:
    return {
        "baseline": baseline,
        "isolation_forest": isolation_forest,
        "selected_detector": selected_detector,
        "selected_threshold": threshold,
        "threshold_calibration_method": method,
        "test_set_used_for_selection": False,
    }


def build_anomaly_evaluation_report(
    final_metrics: dict,
    fault_breakdown: list[dict],
    irradiance_breakdown: list[dict],
    error_analysis: dict,
    metadata: dict,
) -> dict:
    return {
        "evaluation_scope": "FINAL UNTOUCHED TEST RESULT",
        "final_test_metrics": final_metrics,
        "fault_type_breakdown": fault_breakdown,
        "irradiance_breakdown": irradiance_breakdown,
        "error_analysis": error_analysis,
        "model_metadata": metadata,
        "generated_on": date.today().isoformat(),
    }
