from __future__ import annotations

from fastapi import Request

from src.inference.prediction_service import PredictionService


def get_prediction_service(request: Request) -> PredictionService:
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        raise RuntimeError("Prediction service is not initialized")
    return service
