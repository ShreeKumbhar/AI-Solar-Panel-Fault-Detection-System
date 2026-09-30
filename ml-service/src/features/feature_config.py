from __future__ import annotations

BASE_FEATURE_COLUMNS = [
    "vdc1",
    "vdc2",
    "idc1",
    "idc2",
    "irradiance",
    "pv_temperature",
    "power_string_1_w",
    "power_string_2_w",
    "actual_power_w",
]

ENGINEERED_FEATURE_COLUMNS = [
    "power_total_w",
    "power_ratio",
    "voltage_ratio",
    "current_ratio",
    "temperature_irradiance_ratio",
    "irradiance_power_ratio",
    "voltage_balance",
    "current_balance",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS
TARGET_COLUMN = "fault_label"
ID_COLUMN = "sample_index"

FEATURE_SCHEMA = {
    "target_column": TARGET_COLUMN,
    "id_column": ID_COLUMN,
    "base_features": BASE_FEATURE_COLUMNS,
    "engineered_features": ENGINEERED_FEATURE_COLUMNS,
    "feature_columns": FEATURE_COLUMNS,
}

FEATURES = FEATURE_COLUMNS
TARGET = TARGET_COLUMN
