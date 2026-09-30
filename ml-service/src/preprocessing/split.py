from __future__ import annotations

from typing import Tuple

import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataframe by the natural row order without shuffling, preserving chronology."""
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty; cannot create chronological splits.")

    if not (0.0 < train_fraction < 1.0 and 0.0 < validation_fraction < 1.0 and 0.0 < test_fraction < 1.0):
        raise ValueError("All split fractions must be between 0 and 1.")

    total = train_fraction + validation_fraction + test_fraction
    if not (0.999 <= total <= 1.001):
        raise ValueError("Split fractions must sum to 1.0.")

    n_rows = len(df)
    if n_rows < 3:
        raise ValueError("At least three rows are required to create train/validation/test partitions.")

    train_size = int(round(n_rows * train_fraction))
    validation_size = int(round(n_rows * validation_fraction))
    test_size = n_rows - train_size - validation_size

    if train_size < 1 or validation_size < 1 or test_size < 1:
        train_size = max(1, int(n_rows * train_fraction))
        validation_size = max(1, int(n_rows * validation_fraction))
        test_size = n_rows - train_size - validation_size
        if test_size < 1:
            test_size = 1

    train_df = df.iloc[:train_size].copy()
    validation_df = df.iloc[train_size : train_size + validation_size].copy()
    test_df = df.iloc[train_size + validation_size : train_size + validation_size + test_size].copy()

    if len(train_df) == 0 or len(validation_df) == 0 or len(test_df) == 0:
        raise ValueError("One or more splits ended up empty; revise the split fractions.")

    return train_df, validation_df, test_df


def validate_split_lengths(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> bool:
    return len(train_df) > 0 and len(validation_df) > 0 and len(test_df) > 0
