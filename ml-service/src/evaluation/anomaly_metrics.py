from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def calculate_anomaly_metrics(y_true, y_pred) -> dict:
    actual = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=[0, 1],
        average="binary",
        zero_division=0,
    )
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    total = len(actual)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((tp + tn) / total) if total else 0.0,
        "confusion_matrix": matrix.astype(int).tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sample_count": int(total),
        "anomaly_count": int(predicted.sum()),
        "normal_count": int(total - predicted.sum()),
        "true_anomaly_count": int(actual.sum()),
        "true_normal_count": int(total - actual.sum()),
        "anomaly_rate": float(predicted.mean()) if total else 0.0,
        "true_anomaly_rate": float(actual.mean()) if total else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }
