from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.models.fault_classifier import CLASS_LABELS, CLASS_NAMES


def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def plot_class_distribution(distributions: dict[str, list[dict]], output_dir: str | Path) -> Path:
    rows = []
    for split, values in distributions.items():
        for value in values:
            rows.append({"split": split, "class_name": value["class_name"], "count": value["count"]})
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=pd.DataFrame(rows), x="class_name", y="count", hue="split", ax=axis)
    axis.set(xlabel="Fault class", ylabel="Count", title="Fault class distribution")
    axis.tick_params(axis="x", rotation=25)
    return _save(figure, Path(output_dir) / "fault_class_distribution.png")


def plot_confusion_matrix(metrics: dict, output_dir: str | Path, filename: str, normalized: bool = False) -> Path:
    matrix = metrics["confusion_matrix"]
    if normalized:
        import numpy as np

        array = np.asarray(matrix, dtype=float)
        matrix = np.divide(array, array.sum(axis=1, keepdims=True), out=np.zeros_like(array), where=array.sum(axis=1, keepdims=True) != 0)
        fmt = ".2f"
    else:
        fmt = "d"
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap="Blues", cbar=False, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axis)
    axis.set(xlabel="Predicted class", ylabel="True class", title=filename.replace("_", " ").removesuffix(".png"))
    return _save(figure, Path(output_dir) / filename)


def plot_feature_importance(feature_importance: list[dict], output_dir: str | Path) -> Path:
    ordered = sorted(feature_importance, key=lambda item: item["importance"])
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.barh([item["feature"] for item in ordered], [item["importance"] for item in ordered])
    axis.set(xlabel="Importance", ylabel="Feature", title="Fault classifier feature importance")
    return _save(figure, Path(output_dir) / "fault_feature_importance.png")
