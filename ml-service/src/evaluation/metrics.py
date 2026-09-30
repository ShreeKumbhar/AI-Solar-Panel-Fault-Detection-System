from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_regression_metrics(y_true, y_pred) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def calculate_group_metrics(frame: pd.DataFrame, irradiance_column: str = "irradiance") -> list[dict]:
    if frame.empty:
        return []
    values = frame[irradiance_column].astype(float)
    groups = pd.qcut(values, q=3, labels=["low", "medium", "high"], duplicates="drop")
    result = []
    for group_name, group in frame.assign(_irradiance_group=groups).groupby("_irradiance_group", observed=False):
        if len(group) == 0:
            continue
        metrics = calculate_regression_metrics(group["actual_power_w"], group["expected_power_w"])
        result.append({"group": str(group_name), "sample_count": int(len(group)), **metrics})
    return result
