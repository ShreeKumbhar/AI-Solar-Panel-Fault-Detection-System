from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def validate_required_columns(required_columns: Iterable[str], present_columns: Iterable[str]) -> bool:
    """Validate that all required columns are present in a DataFrame or collection."""
    required = set(required_columns)
    present = set(present_columns)
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    return True


def validate_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> bool:
    """Check that all selected columns are numeric or convertible to numeric values."""
    column_list = list(columns)
    validate_required_columns(column_list, df.columns)

    for column in column_list:
        numeric_series = pd.to_numeric(df[column], errors="coerce")
        if numeric_series.isna().sum() != df[column].isna().sum():
            raise ValueError(f"Column '{column}' contains non-numeric values that cannot be parsed.")
    return True


def validate_no_infinite_values(df: pd.DataFrame) -> dict:
    """Return a summary of infinite values across all numeric columns."""
    infinite_counts = {}
    total = 0
    for column in df.columns:
        series = pd.to_numeric(df[column], errors="coerce")
        count = int(np.isinf(series.to_numpy(dtype=float)).sum())
        infinite_counts[column] = count
        total += count
    return {"has_infinite_values": total > 0, "infinite_counts": infinite_counts}


def validate_value_ranges(df: pd.DataFrame, columns: Iterable[str] | None = None) -> dict:
    """Report observed min/max values for numeric columns without imposing arbitrary limits."""
    if columns is None:
        columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    else:
        validate_required_columns(columns, df.columns)

    ranges = {}
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        finite = values[np.isfinite(values.to_numpy(dtype=float))]
        if len(finite) == 0:
            ranges[column] = {"min": None, "max": None}
        else:
            ranges[column] = {"min": float(finite.min()), "max": float(finite.max())}
    return ranges


def validate_fault_labels(df: pd.DataFrame, column: str = "fault_label") -> bool:
    """Check that fault labels are finite and fall within the observed label set."""
    if column not in df.columns:
        raise ValueError(f"Fault label column '{column}' not found in the dataset.")

    labels = pd.to_numeric(df[column], errors="coerce")
    if labels.isna().any():
        raise ValueError(f"Fault label column '{column}' contains missing or invalid values.")

    unique_labels = sorted(np.unique(labels.to_numpy(dtype=float)))
    if unique_labels and (float(unique_labels[0]) < 0 or float(unique_labels[-1]) > 4):
        raise ValueError(
            f"Fault label column '{column}' contains values outside the observed dataset range [0, 4]: {unique_labels}"
        )
    return True


def validate_lengths(left: Iterable[object], right: Iterable[object], label: str = "samples") -> bool:
    """Ensure two sequences contain the same number of rows before combining them."""
    left_len = len(list(left))
    right_len = len(list(right))
    if left_len != right_len:
        raise ValueError(
            f"{label} lengths do not align: left={left_len}, right={right_len}. "
            "Stop and inspect the source arrays before combining them."
        )
    return True
