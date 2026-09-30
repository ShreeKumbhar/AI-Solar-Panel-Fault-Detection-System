# Phase 7: FastAPI ML inference service

This phase exposes the frozen Phase 3 expected-power model, Phase 4 Isolation Forest detector, and Phase 5 fault classifier over HTTP. No model is retrained or redesigned.

## Architecture

Telemetry enters FastAPI, where the service calculates Phase 1 power, runs expected-power inference, constructs causal Phase 4 residual/history features, runs anomaly detection, then runs Phase 5 classification and confidence handling. `panel_id` and `timestamp` remain request context and are never passed to model features.

## Model loading

The application lifespan loads these artifacts once from `MODEL_ARTIFACT_DIR` or the repository `artifacts/` directory:

- `expected_power_model.joblib` and its metadata/schema
- `anomaly_detector.joblib` and its metadata
- `fault_classifier.joblib` and its metadata/schema

Startup fails clearly when artifacts are missing or schemas are incompatible.

## Historical features

History is kept in memory per panel using a trailing window of 10 observations. Histories are never mixed between panels. Requests for a panel must have strictly increasing timezone-aware timestamps. The first observation uses the existing Phase 4 fallback semantics: current residuals, rolling values over the available rows, zero previous-value fields, and zero standard deviation when there is insufficient history. No future observations are used.

## Endpoints

- `GET /api/v1/health` checks service/model readiness.
- `GET /api/v1/model` returns model names, class names, threshold, and versions.
- `POST /api/v1/predict` accepts one telemetry observation.
- `POST /api/v1/predict/batch` accepts up to 1,000 observations and reuses the same single-observation logic.
- `/docs` and `/openapi.json` are provided by FastAPI.

The input fields are `panel_id`, timezone-aware `timestamp`, `vdc1`, `vdc2`, `idc1`, `idc2`, `irradiance`, and `pv_temperature`. Actual power is calculated as `vdc1 * idc1 + vdc2 * idc2`.

## Confidence and severity

Confidence is the maximum Phase 5 class probability and is not a calibrated real-world probability. If Phase 4 marks an observation anomalous and confidence is below the saved Phase 5 validation threshold, the service returns `UNKNOWN_ANOMALY`.

Severity is an application-level deterministic field: `NORMAL` when not anomalous, `CRITICAL` for open/short circuit or absolute deviation at least 50%, and `WARNING` for other anomalies. Severity is not a model output.

## Run locally

```bash
source .venv/bin/activate
python scripts/run_api.py
```

Configuration is read from environment variables. See [.env.example](../.env.example): `MODEL_ARTIFACT_DIR`, `API_HOST`, `API_PORT`, and `LOG_LEVEL`.

## Tests

```bash
pytest -q tests/test_api.py
pytest -q
```

## Limitations

The service is inference-only and uses previously trained models. The public dataset is not 500-panel field data; Phase 6 is a virtual farm simulation. Historical state is process-local memory and is lost on restart or across multiple service instances. Spring Boot, PostgreSQL, React, authentication, streaming, and deployment are later phases.
