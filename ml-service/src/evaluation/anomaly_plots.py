from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def create_anomaly_plots(frame, output_dir: str | Path) -> list[Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    x = frame["sample_index"]
    paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(x, frame["actual_power_w"], label="Actual", linewidth=0.7)
    axis.plot(x, frame["expected_power_w"], label="Expected", linewidth=0.7)
    axis.set(xlabel="Sample index", ylabel="Power (W)", title="Actual vs expected power")
    axis.legend()
    paths.append(_save(figure, output_root / "actual_vs_expected_power.png"))

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(x, frame["deviation_percent"], linewidth=0.7)
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set(xlabel="Sample index", ylabel="Deviation (%)", title="Power deviation over time")
    paths.append(_save(figure, output_root / "power_deviation_over_time.png"))

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(x, frame["anomaly_score"], linewidth=0.7, label="Anomaly score")
    anomalies = frame[frame["anomaly"]]
    axis.scatter(anomalies["sample_index"], anomalies["anomaly_score"], s=5, color="crimson", label="Anomaly")
    axis.set(xlabel="Sample index", ylabel="Normalized anomaly score", title="Anomaly score over time")
    axis.legend()
    paths.append(_save(figure, output_root / "anomaly_score_over_time.png"))

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(frame["residual_w"], bins=80)
    axis.set(xlabel="Residual (W)", ylabel="Count", title="Residual distribution")
    paths.append(_save(figure, output_root / "residual_distribution.png"))

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.scatter(frame["irradiance"], frame["residual_w"], s=3, alpha=0.25)
    axis.set(xlabel="Irradiance", ylabel="Residual (W)", title="Residual vs irradiance")
    paths.append(_save(figure, output_root / "residual_vs_irradiance.png"))

    figure, axis = plt.subplots(figsize=(8, 5))
    sns.histplot(data=frame, x="anomaly_score", hue="is_anomaly_true", bins=50, stat="density", common_norm=False, ax=axis)
    axis.set(title="Anomaly score by normal/fault label", xlabel="Normalized anomaly score")
    paths.append(_save(figure, output_root / "score_distribution_by_label.png"))
    return paths


def create_confusion_matrix_plot(metrics: dict, output_dir: str | Path) -> Path:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set(xlabel="Predicted", ylabel="Actual", title="Anomaly confusion matrix")
    return _save(figure, output_root / "anomaly_confusion_matrix.png")


def _save(figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path
