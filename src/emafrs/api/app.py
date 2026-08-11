"""FastAPI backend -- build brief Section 28: production API.

Endpoints are real: they read/write the actual SQLite DB (emafrs.db) and, for the predict
endpoints, actually run the real training/inference pipeline (emafrs.mlops). Nothing here
returns a canned/example response.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from emafrs.api.middleware import InMemoryRateLimitMiddleware, RequestLoggingMiddleware, SecurityHeadersMiddleware
from emafrs.api.schemas import (
    CounterfactualResponse,
    DataSourceHealth,
    DataSourcesHealthResponse,
    HealthResponse,
    ModelVersionResponse,
    PredictionResponse,
)
from emafrs.db.models import ModelVersion, PredictionRecord
from emafrs.db.session import get_session, init_db

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("emafrs.api")

APP_VERSION = "0.1.0"
DIRECTION_LABELS = {-1: "bearish", 0: "neutral", 1: "bullish"}

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    init_db()
    logger.info("EMAFRS API started (version %s)", APP_VERSION)
    yield


app = FastAPI(
    title="EMAFRS / AMIF API",
    description="Explainable Multimodal Affective Financial Reasoning System -- research reconstruction API.",
    version=APP_VERSION,
    lifespan=_lifespan,
)

_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(InMemoryRateLimitMiddleware, requests_per_window=60, window_seconds=60)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=APP_VERSION)


@app.get("/api/v1/data-sources/health", response_model=DataSourcesHealthResponse)
def data_sources_health(live: bool = Query(False, description="If true, probes each configured source with one real request (costs quota on some providers -- see docs/DATA_SOURCES_AND_APIS.md).")) -> DataSourcesHealthResponse:
    from emafrs.data.health import check_all

    results = check_all(live=live)
    return DataSourcesHealthResponse(
        sources=[
            DataSourceHealth(
                provider=name, configured=h.configured, reachable=h.reachable,
                last_success_at=h.last_success_at, last_error=h.last_error, notes=h.notes,
            )
            for name, h in results.items()
        ],
        checked_live=live,
    )


def _to_prediction_response(record: PredictionRecord) -> PredictionResponse:
    cf = None
    if record.counterfactual_json:
        cf = CounterfactualResponse(**record.counterfactual_json)
    return PredictionResponse(
        id=record.id, ticker=record.ticker, as_of_date=record.as_of_date,
        direction=record.direction, direction_label=DIRECTION_LABELS.get(record.direction, "unknown"),
        confidence=record.confidence, regime=record.regime, regime_name=record.regime_name,
        modality_weights=record.modality_weights_json, top_attributions=record.top_attributions_json,
        counterfactual=cf, manipulation_risk=record.manipulation_risk,
        signal_consistency=record.signal_consistency,
        explanation_text=record.explanation_text, model_version_id=record.model_version_id,
        created_at=record.created_at,
    )


@app.get("/api/v1/predictions/{ticker}/latest", response_model=PredictionResponse)
def get_latest_prediction(ticker: str) -> PredictionResponse:
    """Returns the most recently PERSISTED prediction for `ticker`. Does NOT trigger a
    new training run -- use POST .../refresh for that. Returns 404 if none exists yet
    (never fabricates a placeholder prediction)."""
    session = get_session()
    try:
        record = (
            session.query(PredictionRecord)
            .filter(PredictionRecord.ticker == ticker.upper())
            .order_by(PredictionRecord.created_at.desc())
            .first()
        )
        if record is None:
            raise HTTPException(status_code=404, detail=f"No prediction has been computed yet for {ticker.upper()}. POST /api/v1/predictions/{ticker.upper()}/refresh to compute one.")
        return _to_prediction_response(record)
    finally:
        session.close()


@app.post("/api/v1/predictions/{ticker}/refresh", response_model=PredictionResponse)
def refresh_prediction(ticker: str) -> PredictionResponse:
    """Runs the REAL training + inference + XAI pipeline for `ticker` synchronously and
    persists the result. This is a demo-scale synchronous pattern (training takes roughly
    10-30 seconds on CPU) -- see docs/PRODUCTION_DEPLOYMENT.md for the background-worker
    pattern a higher-traffic deployment should use instead."""
    from emafrs.mlops.training_pipeline import run_training_and_prediction_pipeline

    try:
        record = run_training_and_prediction_pipeline(ticker.upper())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _to_prediction_response(record)


@app.get("/api/v1/predictions/{ticker}/history", response_model=list[PredictionResponse])
def get_prediction_history(ticker: str, limit: int = Query(20, ge=1, le=200)) -> list[PredictionResponse]:
    session = get_session()
    try:
        records = (
            session.query(PredictionRecord)
            .filter(PredictionRecord.ticker == ticker.upper())
            .order_by(PredictionRecord.created_at.desc())
            .limit(limit)
            .all()
        )
        return [_to_prediction_response(r) for r in records]
    finally:
        session.close()


@app.get("/api/v1/model-versions/{ticker}", response_model=list[ModelVersionResponse])
def get_model_versions(ticker: str) -> list[ModelVersionResponse]:
    session = get_session()
    try:
        versions = (
            session.query(ModelVersion)
            .filter(ModelVersion.ticker == ticker.upper())
            .order_by(ModelVersion.trained_at.desc())
            .all()
        )
        return [
            ModelVersionResponse(
                id=v.id, model_type=v.model_type, ticker=v.ticker, metrics=v.metrics_json,
                trained_at=v.trained_at, is_active=v.is_active,
            )
            for v in versions
        ]
    finally:
        session.close()
