"""Persistence layer -- SQLAlchemy ORM models.

Predictions and model-version metadata are persisted so the API can serve the latest
computed prediction/explanation without retraining on every request, and so model
performance is auditable over time (build brief Section 35: model versioning).
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PredictionRecord(Base):
    """One persisted prediction for one ticker at one point in time, including the
    explanation and manipulation risk computed alongside it -- the unit the dashboard and
    API read back."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    as_of_date: Mapped[str] = mapped_column(String(10))  # ISO date the prediction covers
    direction: Mapped[int] = mapped_column(Integer)  # -1, 0, 1
    confidence: Mapped[float] = mapped_column(Float)
    regime: Mapped[int] = mapped_column(Integer)
    regime_name: Mapped[str] = mapped_column(String(16))
    modality_weights_json: Mapped[dict] = mapped_column(JSON)
    top_attributions_json: Mapped[dict] = mapped_column(JSON)
    counterfactual_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    manipulation_risk: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_consistency: Mapped[float | None] = mapped_column(Float, nullable=True)  # real cross-modal agreement, in [-1, 1]; None when <2 modalities contributed evidence (undefined, not "unavailable")
    explanation_text: Mapped[str] = mapped_column(String)
    model_version_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ModelVersion(Base):
    """Metadata for one trained model checkpoint -- build brief Section 35's model
    versioning requirement. `checkpoint_path` points at the actual torch state_dict on
    disk (models/checkpoints/, gitignored); this row is the queryable index over it."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_type: Mapped[str] = mapped_column(String(32))  # "static" or "rama"
    ticker: Mapped[str] = mapped_column(String(16))
    checkpoint_path: Mapped[str] = mapped_column(String)
    metrics_json: Mapped[dict] = mapped_column(JSON)
    modality_columns_json: Mapped[dict] = mapped_column(JSON)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    is_active: Mapped[bool] = mapped_column(default=True)


class DataFreshnessRecord(Base):
    """Snapshot of DAL source health, refreshed by the background scheduler
    (emafrs.mlops) -- backs the dashboard's "API health and data freshness" panel
    (build brief Section 29) without hitting every provider on every page load."""

    __tablename__ = "data_freshness"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(32), index=True)
    configured: Mapped[bool] = mapped_column()
    reachable: Mapped[bool | None] = mapped_column(nullable=True)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(String, nullable=True)
    checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
