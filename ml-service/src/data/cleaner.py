from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_mat_file
from src.data.validator import (
    validate_fault_labels,
    validate_lengths,
    validate_no_infinite_values,
    validate_numeric_columns,
    validate_required_columns,
)


def _flatten_vector(value) -> np.ndarray:
    """Convert MATLAB arrays to a 1D NumPy vector without changing the observed sample order."""
    array = np.asarray(value)
    return np.ravel(array).astype(float, copy=False)


def load_verified_dataframe(elec_path: str | Path, amb_path: str | Path) -> pd.DataFrame:
    """Load and align the actual electrical and ambient signals, then build the verified DataFrame."""
    electrical = load_mat_file(elec_path)
    ambient = load_mat_file(amb_path)

    required_electrical = ["vdc1", "vdc2", "idc1", "idc2"]
    required_ambient = ["irr", "pvt", "f_nv"]
    validate_required_columns(required_electrical, electrical.keys())
    validate_required_columns(required_ambient, ambient.keys())

    df = pd.DataFrame(
        {
            "vdc1": _flatten_vector(electrical["vdc1"]),
            "vdc2": _flatten_vector(electrical["vdc2"]),
            "idc1": _flatten_vector(electrical["idc1"]),
            "idc2": _flatten_vector(electrical["idc2"]),
            "irradiance": _flatten_vector(ambient["irr"]),
            "pv_temperature": _flatten_vector(ambient["pvt"]),
            "fault_label": _flatten_vector(ambient["f_nv"]).astype(int),
        }
    )

    validate_lengths(df["vdc1"], df["vdc2"], "voltage")
    validate_lengths(df["vdc1"], df["idc1"], "current")
    validate_lengths(df["vdc1"], df["irradiance"], "irradiance")
    validate_lengths(df["vdc1"], df["pv_temperature"], "temperature")
    validate_lengths(df["vdc1"], df["fault_label"], "fault labels")

    validate_numeric_columns(df, ["vdc1", "vdc2", "idc1", "idc2", "irradiance", "pv_temperature", "fault_label"])
    validate_fault_labels(df, "fault_label")

    df["power_string_1_w"] = df["vdc1"] * df["idc1"]
    df["power_string_2_w"] = df["vdc2"] * df["idc2"]
    df["actual_power_w"] = df["power_string_1_w"] + df["power_string_2_w"]

    infinite_summary = validate_no_infinite_values(df)
    if infinite_summary["has_infinite_values"]:
        raise ValueError(f"Infinite values detected before cleaning: {infinite_summary['infinite_counts']}")

    return df


def clean_pv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create a conservative cleaned copy while preserving raw measurements and fault labels."""
    cleaned = df.copy()
    numeric_columns = [
        "vdc1",
        "vdc2",
        "idc1",
        "idc2",
        "irradiance",
        "pv_temperature",
        "fault_label",
        "power_string_1_w",
        "power_string_2_w",
        "actual_power_w",
    ]
    validate_required_columns(numeric_columns, cleaned.columns)

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if cleaned[numeric_columns].isna().any().any():
        cleaned = cleaned.copy()
    validate_fault_labels(cleaned, "fault_label")

    return cleaned


def save_clean_dataset(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    processed = df.copy()
    columns = [
        "vdc1",
        "vdc2",
        "idc1",
        "idc2",
        "irradiance",
        "pv_temperature",
        "fault_label",
        "power_string_1_w",
        "power_string_2_w",
        "actual_power_w",
    ]
    present = [col for col in columns if col in processed.columns]
    processed = processed[present]
    processed.to_csv(output_file, index=False)
    return output_file
