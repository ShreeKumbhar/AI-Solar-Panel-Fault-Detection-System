from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.models.fault_classifier import CLASSIFIER_FEATURES


OBSERVATION = {
    "panel_id": "P0103",
    "timestamp": "2026-09-30T10:30:00Z",
    "vdc1": 300.2,
    "vdc2": 299.7,
    "idc1": 0.62,
    "idc2": 0.59,
    "irradiance": 850.0,
    "pv_temperature": 38.5,
}


def test_health_and_model_endpoints():
    with TestClient(create_app()) as client:
        health = client.get("/api/v1/health")
        model = client.get("/api/v1/model")
    assert health.status_code == 200
    assert health.json() == {"status": "UP", "service": "pv-ml-service", "models_loaded": True}
    assert model.status_code == 200
    body = model.json()
    assert body["fault_classes"] == ["NORMAL", "SHORT_CIRCUIT", "DEGRADATION", "OPEN_CIRCUIT", "SHADING"]
    assert body["fault_confidence_threshold"] > 0


def test_single_prediction_uses_real_artifacts():
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/predict", json=OBSERVATION)
    assert response.status_code == 200
    body = response.json()
    assert body["panel_id"] == "P0103"
    assert body["timestamp"] == OBSERVATION["timestamp"]
    assert body["fault_type"] in {"NORMAL", "SHORT_CIRCUIT", "DEGRADATION", "OPEN_CIRCUIT", "SHADING", "UNKNOWN_ANOMALY"}
    assert 0.0 <= body["anomaly_score"] <= 1.0
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["actual_power_w"] == OBSERVATION["vdc1"] * OBSERVATION["idc1"] + OBSERVATION["vdc2"] * OBSERVATION["idc2"]
    assert body["severity"] in {"NORMAL", "WARNING", "CRITICAL"}


def test_batch_preserves_count_and_context():
    second = {**OBSERVATION, "panel_id": "P0104", "timestamp": "2026-09-30T10:31:00Z"}
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/predict/batch", json={"observations": [OBSERVATION, second]})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert [item["panel_id"] for item in predictions] == ["P0103", "P0104"]


def test_invalid_input_is_rejected():
    invalid = {**OBSERVATION, "panel_id": "not-a-panel", "irradiance": "not-a-number"}
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/predict", json=invalid)
    assert response.status_code == 422
    assert response.json()["error"] == "INVALID_INPUT"


def test_panel_histories_are_isolated():
    from src.inference.prediction_service import ModelBundle, PredictionService
    from src.api.schemas import TelemetryObservation

    service = PredictionService(ModelBundle.load("artifacts"))
    first = TelemetryObservation(**OBSERVATION)
    other = TelemetryObservation(**{**OBSERVATION, "panel_id": "P0104"})
    service.predict(first)
    service.predict(other)
    assert len(service._histories["P0103"]) == 1
    assert len(service._histories["P0104"]) == 1


def test_topology_fields_are_not_classifier_features():
    assert not {"panel_id", "row_index", "column_index", "string_id", "fault_label"}.intersection(CLASSIFIER_FEATURES)
