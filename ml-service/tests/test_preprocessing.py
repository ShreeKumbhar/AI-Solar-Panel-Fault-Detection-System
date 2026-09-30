from __future__ import annotations

import pandas as pd

from src.features.engineering import engineer_features
from src.preprocessing.split import chronological_split


def test_feature_engineering_adds_expected_numeric_features():
    df = pd.DataFrame(
        {
            "vdc1": [10.0, 20.0, 30.0],
            "vdc2": [5.0, 8.0, 10.0],
            "idc1": [2.0, 3.0, 4.0],
            "idc2": [1.0, 2.0, 3.0],
            "irradiance": [100.0, 200.0, 300.0],
            "pv_temperature": [20.0, 25.0, 30.0],
            "fault_label": [0, 1, 0],
            "power_string_1_w": [20.0, 60.0, 120.0],
            "power_string_2_w": [5.0, 16.0, 30.0],
            "actual_power_w": [25.0, 76.0, 150.0],
        }
    )

    engineered = engineer_features(df)

    assert "power_total_w" in engineered.columns
    assert "power_ratio" in engineered.columns
    assert "voltage_ratio" in engineered.columns
    assert "temperature_irradiance_ratio" in engineered.columns
    assert "sample_index" in engineered.columns
    assert engineered["power_total_w"].tolist() == [25.0, 76.0, 150.0]


def test_chronological_split_preserves_order_and_partition_sizes():
    df = pd.DataFrame(
        {
            "sample_index": list(range(10)),
            "fault_label": [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "vdc1": [float(i) for i in range(10)],
        }
    )

    train, validation, test = chronological_split(df, train_fraction=0.6, validation_fraction=0.2, test_fraction=0.2)

    assert len(train) == 6
    assert len(validation) == 2
    assert len(test) == 2
    assert train["sample_index"].tolist() == list(range(6))
    assert validation["sample_index"].tolist() == [6, 7]
    assert test["sample_index"].tolist() == [8, 9]
