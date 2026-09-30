from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.fault_classifier import (
    CLASS_NAMES,
    CLASS_LABELS,
    CLASSIFIER_FEATURES,
    FaultClassifier,
    apply_unknown_anomaly_rule,
    build_classification_features,
    calibrate_confidence_threshold,
)
from src.evaluation.fault_metrics import calculate_fault_metrics, class_distribution


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_index": range(10),
            "vdc1": np.linspace(100, 109, 10),
            "vdc2": np.linspace(98, 107, 10),
            "idc1": np.linspace(1, 2, 10),
            "idc2": np.linspace(0.8, 1.8, 10),
            "irradiance": np.linspace(100, 900, 10),
            "pv_temperature": np.linspace(20, 29, 10),
            "actual_power_w": np.linspace(180, 260, 10),
            "expected_power_w": np.linspace(190, 250, 10),
            "residual_w": np.linspace(-10, 10, 10),
            "absolute_error_w": np.linspace(10, 2, 10),
            "deviation_percent": np.linspace(-5, 5, 10),
            "rolling_mean_residual": np.linspace(-10, 5, 10),
            "rolling_std_residual": np.ones(10),
            "rolling_mean_absolute_error": np.linspace(10, 3, 10),
            "rolling_mean_deviation": np.linspace(-5, 3, 10),
            "previous_residual": np.linspace(-10, 8, 10),
            "previous_deviation": np.linspace(-5, 4, 10),
            "deviation_trend": np.ones(10),
            "anomaly_score": np.linspace(0.1, 0.9, 10),
            "anomaly": [False] * 5 + [True] * 5,
            "fault_label": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        }
    )


def test_mapping_and_leakage_safe_features():
    frame = build_classification_features(sample_frame())
    assert CLASS_LABELS == {0: "NORMAL", 1: "SHORT_CIRCUIT", 2: "DEGRADATION", 3: "OPEN_CIRCUIT", 4: "SHADING"}
    assert CLASS_NAMES == list(CLASS_LABELS.values())
    assert "fault_label" not in CLASSIFIER_FEATURES
    assert not {"panel_id", "row_index", "column_index"}.intersection(CLASSIFIER_FEATURES)
    assert not frame[CLASSIFIER_FEATURES].isna().any().any()
    numeric = frame[CLASSIFIER_FEATURES].astype(float)
    assert np.isfinite(numeric.to_numpy()).all()


def test_classifier_probabilities_and_unknown_rule():
    frame = build_classification_features(sample_frame())
    classifier = FaultClassifier(random_state=42)
    classifier.fit(frame[CLASSIFIER_FEATURES], frame["fault_label"])
    probabilities = classifier.predict_proba(frame[CLASSIFIER_FEATURES])
    predictions = classifier.predict(frame[CLASSIFIER_FEATURES])
    assert probabilities.shape == (len(frame), 5)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(predictions).issubset(set(CLASS_LABELS))
    assert np.allclose(classifier.confidence(probabilities), probabilities.max(axis=1))
    output = apply_unknown_anomaly_rule(predictions, probabilities, np.ones(len(frame), dtype=bool), threshold=0.99)
    assert output["unknown_anomaly"].any()


def test_confidence_threshold_and_chronology():
    probabilities = np.array([[0.9, 0.05, 0.02, 0.02, 0.01], [0.2, 0.2, 0.2, 0.2, 0.2]])
    labels = np.array([0, 4])
    anomalies = np.array([False, True])
    threshold = calibrate_confidence_threshold(probabilities, labels, anomalies)
    assert 0.2 <= threshold <= 0.9
    assert sample_frame()["sample_index"].is_monotonic_increasing


def test_classifier_save_load_and_metrics(tmp_path):
    frame = build_classification_features(sample_frame())
    classifier = FaultClassifier(random_state=42)
    classifier.fit(frame[CLASSIFIER_FEATURES], frame["fault_label"])
    path = classifier.save(tmp_path / "fault_classifier.joblib")
    loaded = FaultClassifier.load(path)
    assert np.array_equal(
        classifier.predict(frame[CLASSIFIER_FEATURES]),
        loaded.predict(frame[CLASSIFIER_FEATURES]),
    )
    metrics = calculate_fault_metrics(frame["fault_label"], classifier.predict(frame[CLASSIFIER_FEATURES]))
    assert len(metrics["confusion_matrix"]) == 5
    assert len(metrics["confusion_matrix"][0]) == 5


def test_class_distribution_reports_all_classes():
    distribution = class_distribution(sample_frame()["fault_label"])
    assert [row["class_name"] for row in distribution] == CLASS_NAMES
    assert sum(row["count"] for row in distribution) == len(sample_frame())
