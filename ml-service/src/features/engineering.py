from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray, eps: float = 1e-9) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return numerator / (denominator.replace(0, np.nan) + eps).fillna(0.0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add consistent, non-leaky numerical features while preserving the original PV signals."""
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty; feature engineering requires at least one sample.")

    engineered = df.copy()

    if "sample_index" not in engineered.columns:
        engineered["sample_index"] = np.arange(len(engineered), dtype=int)

    for column in ["power_string_1_w", "power_string_2_w", "actual_power_w"]:
        if column not in engineered.columns:
            if column == "power_string_1_w":
                engineered[column] = engineered.get("vdc1", 0.0) * engineered.get("idc1", 0.0)
            elif column == "power_string_2_w":
                engineered[column] = engineered.get("vdc2", 0.0) * engineered.get("idc2", 0.0)
            else:
                engineered[column] = engineered.get("power_string_1_w", 0.0) + engineered.get("power_string_2_w", 0.0)

    engineered["power_total_w"] = engineered["actual_power_w"].astype(float)
    engineered["power_ratio"] = _safe_divide(engineered["power_string_1_w"], engineered["power_string_2_w"])
    engineered["voltage_ratio"] = _safe_divide(engineered["vdc1"], engineered["vdc2"])
    engineered["current_ratio"] = _safe_divide(engineered["idc1"], engineered["idc2"])
    engineered["temperature_irradiance_ratio"] = _safe_divide(engineered["pv_temperature"], engineered["irradiance"])
    engineered["irradiance_power_ratio"] = _safe_divide(engineered["actual_power_w"], engineered["irradiance"])
    engineered["voltage_balance"] = (
        pd.Series(np.abs(engineered["vdc1"] - engineered["vdc2"]), index=engineered.index)
        / (pd.Series(np.abs(engineered["vdc1"]) + np.abs(engineered["vdc2"]) + 1e-9, index=engineered.index))
    )
    engineered["current_balance"] = (
        pd.Series(np.abs(engineered["idc1"] - engineered["idc2"]), index=engineered.index)
        / (pd.Series(np.abs(engineered["idc1"]) + np.abs(engineered["idc2"]) + 1e-9, index=engineered.index))
    )

    for column in [
        "power_total_w",
        "power_ratio",
        "voltage_ratio",
        "current_ratio",
        "temperature_irradiance_ratio",
        "irradiance_power_ratio",
        "voltage_balance",
        "current_balance",
    ]:
        engineered[column] = pd.to_numeric(engineered[column], errors="coerce").fillna(0.0)

    return engineered
