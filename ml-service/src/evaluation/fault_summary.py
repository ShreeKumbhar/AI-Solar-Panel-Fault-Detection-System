from __future__ import annotations

from pathlib import Path


def write_fault_summary(path: str | Path, comparison: dict, evaluation: dict) -> Path:
    destination = Path(path)
    selected = comparison["selected_model"]
    metrics = evaluation["final_test_metrics"]
    lines = [
        "# Phase 5: fault classification",
        "",
        "This phase predicts NORMAL, SHORT_CIRCUIT, DEGRADATION, OPEN_CIRCUIT, or SHADING after the expected-power and anomaly stages. It does not implement the virtual farm or application layers.",
        "",
        "## Target and labels",
        "",
        "The verified public-dataset labels are 0 Normal, 1 Short-Circuit, 2 Degradation, 3 Open-Circuit, and 4 Shadowing. The application name for label 4 is SHADING. `fault_label` is the target only and is excluded from features.",
        "",
        "## Features and leakage prevention",
        "",
        "Features combine raw PV measurements with Phase 3 expected power and Phase 4 causal residual/anomaly context. No fault label, panel identifier, row/column identifier, or future-derived feature is used. The combined chronological anomaly stream is rebuilt before partitioning so causal rolling context continues across split boundaries.",
        "",
        "## Model selection",
        "",
        f"Random Forest and XGBoost are compared using validation macro F1, with macro recall as the tie-breaker. The selected model was `{selected}`. Test data is not used for selection. Class weights/sample weights address imbalance for the training data only.",
        "",
        "## Uncertainty",
        "",
        f"The validation-selected confidence threshold is `{comparison['selected_confidence_threshold']}`. A Phase 4 anomaly with maximum class probability below this threshold is emitted as `UNKNOWN_ANOMALY`. These probabilities are model estimates, not guaranteed calibrated probabilities.",
        "",
        "## Final test result",
        "",
        f"Accuracy: {metrics['accuracy']:.4f}; macro F1: {metrics['f1_macro']:.4f}; weighted F1: {metrics['f1_weighted']:.4f}.",
        "",
        "## Limitations",
        "",
        "The chronological training split contains no Short-Circuit rows, so that class is unseen during fitting; validation/test contain it and the resulting metric must be interpreted accordingly. The public dataset represents one two-string PV plant, not 500 physical panels. Low irradiance and overlapping shading/degradation signatures remain important limitations. UNKNOWN_ANOMALY is an application uncertainty output, not a sixth training class.",
        "",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
