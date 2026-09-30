from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.models.fault_classifier import CLASS_LABELS, CLASS_NAMES


def calculate_fault_metrics(y_true, predictions) -> dict:
    labels = list(CLASS_LABELS)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_macro": float(precision_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, predictions, labels=labels, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_true, predictions, labels=labels, target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
        "labels": labels,
        "class_names": CLASS_NAMES,
    }


def class_distribution(y, labels: dict[int, str] = CLASS_LABELS) -> list[dict]:
    values = np.asarray(y, dtype=int)
    total = len(values)
    return [
        {"label": int(label), "class_name": name, "count": int((values == label).sum()), "percentage": float((values == label).mean() * 100) if total else 0.0}
        for label, name in labels.items()
    ]
