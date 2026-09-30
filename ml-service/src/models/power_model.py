from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

POWER_TARGET = "actual_power_w"
POWER_FEATURES = ["vdc1", "vdc2", "idc1", "idc2", "irradiance", "pv_temperature"]
POWER_MODEL_COLUMNS = ["sample_index", *POWER_FEATURES, POWER_TARGET]
NORMAL_LABEL = 0

EXCLUDED_FEATURES = {
    "fault_label": "label information; selecting normal rows only is part of data preparation",
    "power_string_1_w": "direct component of actual_power_w",
    "power_string_2_w": "direct component of actual_power_w",
    "actual_power_w": "the prediction target",
    "power_total_w": "duplicate of actual_power_w",
    "power_ratio": "derived from power_string_1_w and power_string_2_w",
    "irradiance_power_ratio": "derived directly from actual_power_w",
    "sample_index": "ordering identifier, not an operating condition",
    "voltage_ratio": "derived engineered ratio excluded for a conservative leakage policy",
    "current_ratio": "derived engineered ratio excluded for a conservative leakage policy",
    "voltage_balance": "derived engineered ratio excluded for a conservative leakage policy",
    "current_balance": "derived engineered ratio excluded for a conservative leakage policy",
    "temperature_irradiance_ratio": "derived engineered ratio excluded for a conservative leakage policy",
}


def load_power_dataset(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Power model input not found: {source}")
    return pd.read_csv(source)


def validate_power_features(df: pd.DataFrame, features: list[str] = POWER_FEATURES, target: str = POWER_TARGET) -> None:
    required = [*features, target]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Power model dataset is missing required columns: {missing}")
    if target in features:
        raise ValueError(f"Target column '{target}' must not be included in model features.")
    numeric = df[required].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Power model input contains NaN or infinite values.")


def build_power_model_datasets(partitions: Mapping[str, pd.DataFrame], normal_label: int = NORMAL_LABEL) -> dict:
    result: dict[str, pd.DataFrame | list[str] | str] = {
        "features": list(POWER_FEATURES),
        "target": POWER_TARGET,
    }
    for split_name in ["train", "validation", "test"]:
        if split_name not in partitions:
            raise ValueError(f"Missing required partition: {split_name}")
        partition = partitions[split_name].copy()
        if "fault_label" not in partition.columns:
            raise ValueError("Power model preparation requires fault_label to select normal samples.")
        normal = partition.loc[partition["fault_label"] == normal_label].copy()
        if "sample_index" not in normal.columns:
            normal.insert(0, "sample_index", np.arange(len(normal), dtype=int))
        normal = normal[["sample_index", *POWER_FEATURES, POWER_TARGET]]
        validate_power_features(normal)
        result[split_name] = normal.reset_index(drop=True)
    return result


def prepare_power_datasets(
    partition_dir: str | Path,
    output_dir: str | Path,
    normal_label: int = NORMAL_LABEL,
) -> dict[str, Path]:
    partition_root = Path(partition_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    partitions = {
        split_name: load_power_dataset(partition_root / f"{split_name}.csv")
        for split_name in ["train", "validation", "test"]
    }
    datasets = build_power_model_datasets(partitions, normal_label=normal_label)
    paths: dict[str, Path] = {}
    for split_name in ["train", "validation", "test"]:
        path = output_root / f"{split_name}_power.csv"
        datasets[split_name].to_csv(path, index=False)
        paths[split_name] = path
    return paths
