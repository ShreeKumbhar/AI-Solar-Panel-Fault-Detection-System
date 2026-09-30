from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TelemetryObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    panel_id: str = Field(pattern=r"^P\d{4}$")
    timestamp: datetime
    vdc1: float = Field(ge=0.0)
    vdc2: float = Field(ge=0.0)
    idc1: float = Field(ge=0.0)
    idc2: float = Field(ge=0.0)
    irradiance: float = Field(ge=0.0)
    pv_temperature: float = Field(ge=-50.0, le=100.0)

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone offset")
        return value

    @field_validator("vdc1", "vdc2", "idc1", "idc2", "irradiance", "pv_temperature")
    @classmethod
    def values_must_be_finite(cls, value: float) -> float:
        if not (-float("inf") < value < float("inf")):
            raise ValueError("telemetry values must be finite")
        return value


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observations: list[TelemetryObservation] = Field(min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    panel_id: str
    timestamp: datetime
    actual_power_w: float
    expected_power_w: float
    deviation_w: float
    deviation_percent: float
    anomaly: bool
    anomaly_score: float
    fault_type: str
    confidence: float
    severity: Literal["NORMAL", "WARNING", "CRITICAL"]
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    service: str
    models_loaded: bool


class ModelInfoResponse(BaseModel):
    expected_power_model: str
    anomaly_model: str
    fault_classifier: str
    fault_classes: list[str]
    fault_confidence_threshold: float
    model_versions: dict[str, str]
