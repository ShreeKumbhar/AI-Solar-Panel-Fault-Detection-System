from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.inference.prediction_service import ModelBundle, PredictionService

LOGGER = logging.getLogger("pv_ml_service")


def _artifact_dir() -> Path:
    configured = os.getenv("MODEL_ARTIFACT_DIR")
    return Path(configured) if configured else Path(__file__).resolve().parents[2] / "artifacts"


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact_dir = _artifact_dir()
    LOGGER.info("Loading frozen ML artifacts from %s", artifact_dir)
    bundle = ModelBundle.load(artifact_dir)
    app.state.prediction_service = PredictionService(bundle)
    LOGGER.info("Frozen ML artifacts loaded")
    yield
    app.state.prediction_service = None


def create_app() -> FastAPI:
    application = FastAPI(
        title="PV ML Inference Service",
        version="1.0.0",
        description="Inference-only API for the frozen Phase 3-5 PV ML pipeline.",
        lifespan=lifespan,
    )
    application.include_router(router)

    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"error": "INVALID_INPUT", "message": "Request validation failed", "details": exc.errors()})

    @application.exception_handler(FileNotFoundError)
    async def artifact_error_handler(request: Request, exc: FileNotFoundError):
        LOGGER.error("Required model artifacts are unavailable")
        return JSONResponse(status_code=503, content={"error": "MODEL_ARTIFACTS_UNAVAILABLE", "message": "Required ML model artifacts are unavailable"})

    @application.exception_handler(ValueError)
    async def input_value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={"error": "INVALID_INPUT", "message": str(exc)})

    @application.exception_handler(Exception)
    async def internal_error_handler(request: Request, exc: Exception):
        LOGGER.exception("Unhandled inference error")
        return JSONResponse(status_code=500, content={"error": "INFERENCE_ERROR", "message": "The ML inference request could not be completed"})

    return application


app = create_app()
