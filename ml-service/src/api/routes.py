from __future__ import annotations

import logging
from uuid import uuid4
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from src.api.dependencies import get_prediction_service
from src.api.schemas import BatchPredictionRequest, BatchPredictionResponse, HealthResponse, ModelInfoResponse, PredictionResponse, TelemetryObservation
from src.inference.prediction_service import PredictionService

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
def health(service: Annotated[PredictionService, Depends(get_prediction_service)]) -> HealthResponse:
    return HealthResponse(status="UP", service="pv-ml-service", models_loaded=service.bundle is not None)


@router.get("/model", response_model=ModelInfoResponse)
def model_info(service: Annotated[PredictionService, Depends(get_prediction_service)]) -> ModelInfoResponse:
    return ModelInfoResponse(**service.model_info())


@router.post("/predict", response_model=PredictionResponse)
def predict(request: Request, observation: TelemetryObservation, service: Annotated[PredictionService, Depends(get_prediction_service)]) -> PredictionResponse:
    LOGGER.info("prediction request_id=%s panel_id=%s", request.state.request_id, observation.panel_id)
    return PredictionResponse(**service.predict(observation))


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: Request, payload: BatchPredictionRequest, service: Annotated[PredictionService, Depends(get_prediction_service)]) -> BatchPredictionResponse:
    LOGGER.info("batch prediction request_id=%s count=%s", request.state.request_id, len(payload.observations))
    return BatchPredictionResponse(predictions=service.predict_batch(payload.observations))
