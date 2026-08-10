"""Common provenance schema every Data Acquisition Layer client returns.

Every record produced anywhere in the DAL carries where it came from, when it was
fetched, and whether it is stale — per the build brief's requirement that "every data
object must contain: timestamp, source, freshness, provenance" and that stale data must
never be silently treated as live.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class SourceRecord:
    """One unit of data pulled from an external provider."""

    provider: str
    """Short provider id, e.g. 'yfinance', 'alphavantage', 'newsapi', 'youtube', 'fred', 'sec_edgar'."""

    endpoint: str
    """Which logical endpoint/method produced this, e.g. 'daily_ohlcv', 'company_news'."""

    fetched_at: datetime
    """UTC time this record was retrieved by this system (not the data's own timestamp)."""

    observation_time: Optional[datetime]
    """UTC time the underlying observation actually refers to, if the provider gives one.
    None for reference/static data where no observation timestamp applies."""

    entity: Optional[str]
    """Ticker / CIK / query term this record is about, if applicable."""

    fields: dict[str, Any]
    """The actual payload, provider-native field names preserved where practical."""

    license_note: str
    """One-line usage constraint (attribution, non-commercial, redistribution limits)."""

    cache_hit: bool = False
    """True if served from local cache rather than a live request this call."""

    def age_seconds(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now(timezone.utc)
        return (now - self.fetched_at).total_seconds()

    def is_stale(self, max_age_seconds: float, now: Optional[datetime] = None) -> bool:
        return self.age_seconds(now) > max_age_seconds


@dataclass
class SourceHealth:
    """Health/freshness status for one configured data source, surfaced on the dashboard."""

    provider: str
    configured: bool
    reachable: Optional[bool] = None
    last_success_at: Optional[datetime] = None
    last_error: Optional[str] = None
    notes: list[str] = field(default_factory=list)
