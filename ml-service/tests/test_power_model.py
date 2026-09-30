from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.evaluation.metrics import calculate_regression_metrics
from src.models.model_io import load_model, save_model
from src.models.power_model import (
    POWER_TARGET,
    build_power_model_datasets,
    validate_power_features,
)


def sample_power_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_index": range(8),
            "vdc1": np.linspace(10, 17, 8),
            "vdc2": np.linspace(9, 16, 8),
            "idc1": np.linspace(1, 2, 8),
            "idc2": np.linspace(0.8, 1.8, 8),
            "irradiance": np.linspace(100, 800, 8),
            "pv_temperature": np.linspace(20, 27, 8),
            "power_string_1_w": np.linspace(10, 34, 8),
            "power_string_2_w": np.linspace(8, 29, 8),
            "actual_power_w": np.linspace(18, 63, 8),
            "fault_label": [0, 1, 0, 0, 2, 0, 0, 4],
        }
    )


def test_power_target_and_leakage_safe_features():
    datasets = build_power_model_datasets(
        {"train": sample_power_frame(), "validation": sample_power_frame(), "test": sample_power_frame()}
    )
    features = datasets["features"]
    assert datasets["target"] == POWER_TARGET
    assert POWER_TARGET not in features
    assert "power_string_1_w" not in features
    assert "power_string_2_w" not in features
    assert "actual_power_w" not in features
    assert "irradiance_power_ratio" not in features
    assert features == ["vdc1", "vdc2", "idc1", "idc2", "irradiance", "pv_temperature"]


def test_healthy_rows_preserve_order_and_expected_columns():
    datasets = build_power_model_datasets(
        {"train": sample_power_frame(), "validation": sample_power_frame(), "test": sample_power_frame()}
    )
    train = datasets["train"]
    assert train["sample_index"].tolist() == [0, 2, 3, 5, 6]
    assert list(train.columns) == ["sample_index", *datasets["features"], POWER_TARGET]
    validate_power_features(train, datasets["features"], POWER_TARGET)


def test_regression_metrics_are_correct():
    metrics = calculate_regression_metrics([1.0, 2.0, 4.0], [1.0, 4.0, 5.0])
    assert np.isclose(metrics["mae"], 1.0)
    assert np.isclose(metrics["rmse"], np.sqrt(5 / 3))
    assert np.isclose(metrics["r2"], 1 - (5 / (14 / 3)))


def test_small_model_trains_predicts_and_round_trips(tmp_path):
    datasets = build_power_model_datasets(
        {"train": sample_power_frame(), "validation": sample_power_frame(), "test": sample_power_frame()}
    )
    train = datasets["train"]
    model = RandomForestRegressor(n_estimators=3, random_state=42)
    model.fit(train[datasets["features"]], train[datasets["target"]])
    predictions = model.predict(train[datasets["features"]])
    assert len(predictions) == len(train)
    assert np.isfinite(predictions).all()

    path = save_model(model, tmp_path / "expected_power_model.joblib")
    loaded = load_model(path)
    assert np.allclose(loaded.predict(train[datasets["features"]]), predictions)
