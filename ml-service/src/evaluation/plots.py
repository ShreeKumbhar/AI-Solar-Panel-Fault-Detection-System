from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def create_power_plots(predictions: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    plots: list[Path] = []

    figures = [
        ("actual_vs_expected_over_time.png", lambda ax: (ax.plot(predictions["sample_index"], predictions["actual_power_w"], label="Actual", linewidth=0.8), ax.plot(predictions["sample_index"], predictions["expected_power_w"], label="Expected", linewidth=0.8), ax.set_xlabel("Sample index"), ax.set_ylabel("Power (W)"), ax.legend())),
        ("actual_vs_expected_scatter.png", lambda ax: (ax.scatter(predictions["actual_power_w"], predictions["expected_power_w"], s=4, alpha=0.25), ax.set_xlabel("Actual power (W)"), ax.set_ylabel("Expected power (W)"))),
        ("residual_distribution.png", lambda ax: ax.hist(predictions["residual_w"], bins=80)),
        ("residual_vs_irradiance.png", lambda ax: ax.scatter(predictions["irradiance"], predictions["residual_w"], s=4, alpha=0.25)),
    ]
    for filename, draw in figures:
        figure, axis = plt.subplots(figsize=(10, 5))
        draw(axis)
        figure.tight_layout()
        path = output_root / filename
        figure.savefig(path, dpi=140)
        plt.close(figure)
        plots.append(path)
    return plots


def create_feature_importance_plot(feature_importance: list[dict], output_dir: str | Path) -> Path:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    importance = pd.DataFrame(feature_importance).sort_values("importance")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(importance["feature"], importance["importance"])
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    figure.tight_layout()
    path = output_root / "power_model_feature_importance.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path
