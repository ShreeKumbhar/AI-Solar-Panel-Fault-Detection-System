from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.anomaly_metrics import calculate_anomaly_metrics
from src.models.anomaly_detector import (
    ANOMALY_FEATURES,
    IsolationForestAnomalyDetector,
    build_anomaly_features,
    calculate_power_deviation,
    calibrate_threshold,
)


def small_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_index": range(6),
            "actual_power_w": [176.0, 200.0, 210.0, 180.0, 215.0, 220.0],
            "expected_power_w": [292.0, 205.0, 208.0, 200.0, 215.0, 220.0],
            "irradiance": [400.0, 410.0, 420.0, 430.0, 440.0, 450.0],
            "pv_temperature": [25.0] * 6,
            "fault_label": [1, 0, 0, 1, 0, 0],
        }
    )


def test_power_deviation_and_zero_expected_handling():
    result = calculate_power_deviation(pd.Series([176.0, 10.0]), pd.Series([292.0, 0.0]))
    assert result["residual_w"].tolist() == [-116.0, 10.0]
    assert np.isclose(result.loc[0, "deviation_percent"], -39.7260274)
    assert result.loc[1, "deviation_percent"] == 0.0


def test_historical_features_use_current_or_past_only():
    frame = build_anomaly_features(small_frame(), rolling_window=3)
    assert frame.loc[0, "previous_residual"] == 0.0
    assert frame.loc[0, "rolling_mean_residual"] == frame.loc[0, "residual_w"]
    assert frame.loc[1, "rolling_mean_residual"] == np.mean(frame.loc[:1, "residual_w"])
    assert frame[ANOMALY_FEATURES].isna().sum().sum() == 0
    assert np.isfinite(frame[ANOMALY_FEATURES].to_numpy()).all()


def test_detector_trains_scores_and_predicts_binary_output():
    frame = build_anomaly_features(small_frame(), rolling_window=3)
    detector = IsolationForestAnomalyDetector(n_estimators=20, random_state=42)
    detector.fit(frame[ANOMALY_FEATURES])
    scores = detector.calculate_anomaly_score(frame[ANOMALY_FEATURES])
    predictions = detector.predict_anomaly(frame[ANOMALY_FEATURES], threshold=0.5)
    assert len(scores) == len(frame)
    assert len(predictions) == len(frame)
    assert np.isfinite(scores).all()
    assert set(predictions.tolist()).issubset({False, True})


def test_detector_save_load_preserves_scores(tmp_path):
    frame = build_anomaly_features(small_frame(), rolling_window=3)
    detector = IsolationForestAnomalyDetector(n_estimators=20, random_state=42)
    detector.fit(frame[ANOMALY_FEATURES])
    detector.threshold = 0.5
    path = detector.save(tmp_path / "anomaly_detector.joblib")
    loaded = IsolationForestAnomalyDetector.load(path)
    assert np.allclose(
        detector.calculate_anomaly_score(frame[ANOMALY_FEATURES]),
        loaded.calculate_anomaly_score(frame[ANOMALY_FEATURES]),
    )


def test_detector_feature_schema_excludes_labels_and_identifiers():
    forbidden = {"fault_label", "panel_id", "row_index", "column_index", "actual_power_w", "expected_power_w"}
    assert forbidden.isdisjoint(ANOMALY_FEATURES)


def test_threshold_and_metrics_are_deterministic():
    scores = np.array([0.05, 0.1, 0.7, 0.9])
    labels = np.array([0, 0, 1, 1])
    threshold = calibrate_threshold(scores, labels)
    assert 0.1 <= threshold <= 0.9
    metrics = calculate_anomaly_metrics(labels, scores >= threshold)
    assert metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"] == 4
    assert len(metrics["confusion_matrix"]) == 2
    assert len(metrics["confusion_matrix"][0]) == 2
