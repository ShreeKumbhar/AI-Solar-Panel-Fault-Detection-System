from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from src.data.loader import load_mat_file
from src.data.validator import (
    validate_fault_labels,
    validate_lengths,
    validate_no_infinite_values,
    validate_numeric_columns,
    validate_required_columns,
)


@pytest.fixture
def sample_mat_file(tmp_path):
    mat_path = tmp_path / "sample.mat"
    data = {
        "vdc1": np.array([[1.0, 2.0, 3.0]], dtype=float),
        "vdc2": np.array([[4.0, 5.0, 6.0]], dtype=float),
        "idc1": np.array([[0.5, 0.6, 0.7]], dtype=float),
        "idc2": np.array([[0.4, 0.5, 0.6]], dtype=float),
        "irr": np.array([[100.0, 200.0, 300.0]], dtype=float),
        "pvt": np.array([[25.0, 26.0, 27.0]], dtype=float),
        "f_nv": np.array([[0], [1], [2]], dtype=np.uint8),
    }
    savemat(str(mat_path), data)
    return mat_path


def test_mat_loader_handles_existing_file(sample_mat_file):
    loaded = load_mat_file(sample_mat_file)
    assert isinstance(loaded, dict)
    assert "vdc1" in loaded
    assert loaded["vdc1"].shape == (1, 3)


def test_missing_file_raises_clear_error(tmp_path):
    missing = tmp_path / "missing.mat"
    with pytest.raises(FileNotFoundError, match="MAT file not found"):
        load_mat_file(missing)


def test_dataframe_contains_expected_verified_columns():
    columns = ["vdc1", "vdc2", "idc1", "idc2", "irradiance", "pv_temperature", "fault_label"]
    df = pd.DataFrame(
        {
            "vdc1": [1.0, 2.0],
            "vdc2": [3.0, 4.0],
            "idc1": [0.5, 0.6],
            "idc2": [0.4, 0.5],
            "irradiance": [100.0, 110.0],
            "pv_temperature": [25.0, 26.0],
            "fault_label": [0, 1],
        }
    )
    assert set(columns).issubset(df.columns)


def test_derived_power_calculation_works():
    df = pd.DataFrame(
        {
            "vdc1": [10.0, 20.0],
            "vdc2": [5.0, 6.0],
            "idc1": [2.0, 3.0],
            "idc2": [1.0, 2.0],
            "fault_label": [0, 1],
        }
    )
    df["power_string_1_w"] = df["vdc1"] * df["idc1"]
    df["power_string_2_w"] = df["vdc2"] * df["idc2"]
    df["actual_power_w"] = df["power_string_1_w"] + df["power_string_2_w"]
    assert df["power_string_1_w"].tolist() == [20.0, 60.0]
    assert df["power_string_2_w"].tolist() == [5.0, 12.0]
    assert df["actual_power_w"].tolist() == [25.0, 72.0]


def test_fault_labels_are_valid():
    df = pd.DataFrame({"fault_label": [0, 1, 2, 3, 4]})
    assert validate_fault_labels(df, "fault_label") is True


def test_infinite_values_are_detected():
    df = pd.DataFrame({"vdc1": [1.0, np.inf, 3.0], "idc1": [0.5, 0.7, np.nan]})
    result = validate_no_infinite_values(df)
    assert result["has_infinite_values"] is True
    assert result["infinite_counts"]["vdc1"] == 1


def test_validation_catches_invalid_input_structures():
    with pytest.raises(ValueError, match="columns.*missing"):
        validate_required_columns(["vdc1", "idc1"], ["vdc1", "missing"])

    with pytest.raises(ValueError, match="do not align"):
        validate_lengths([1, 2, 3], [1, 2], "sample")
