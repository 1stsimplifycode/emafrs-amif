"""Pydantic response models for the FastAPI backend. Every field here is populated from
a real computed/persisted value -- no field defaults to a placeholder that could be
mistaken for a real result.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str


class DataSourceHealth(BaseModel):
    provider: str
    configured: bool
    reachable: bool | None = None
    last_success_at: datetime | None = None
    last_error: str | None = None
    notes: list[str] = []


class DataSourcesHealthResponse(BaseModel):
    sources: list[DataSourceHealth]
    checked_live: bool


class CounterfactualResponse(BaseModel):
    feature_name: str
    delta: float | None
    original_value: float
    counterfactual_value: float | None


class PredictionResponse(BaseModel):
    id: int
    ticker: str
    as_of_date: str
    direction: int
    direction_label: str
    confidence: float
    regime: int
    regime_name: str
    modality_weights: dict[str, float]
    top_attributions: dict[str, float]
    counterfactual: CounterfactualResponse | None
    manipulation_risk: float | None
    signal_consistency: float | None
    explanation_text: str
    model_version_id: int | None
    created_at: datetime


class ModelVersionResponse(BaseModel):
    id: int
    model_type: str
    ticker: str
    metrics: dict
    trained_at: datetime
    is_active: bool
