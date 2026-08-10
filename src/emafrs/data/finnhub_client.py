"""Finnhub client: real-time quote and company news (free tier).

Free tier: 60 API calls/minute. Candle/OHLCV endpoints are restricted on the free tier for
most exchanges as of this writing, so Finnhub is used here for quote snapshots and
company news, not as a primary OHLCV source (yfinance/Alpha Vantage cover that).
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from emafrs.data.base import BaseAPIClient
from emafrs.data.config import CONFIG
from emafrs.data.schema import SourceRecord

_BASE_URL = "https://finnhub.io/api/v1"
LICENSE_NOTE = "Finnhub free tier: 60 calls/minute. See finnhub.io/terms-of-service."


class FinnhubClient(BaseAPIClient):
    provider = "finnhub"

    def __init__(self):
        if not CONFIG.finnhub_api_key:
            raise RuntimeError("FINNHUB_API_KEY not configured")
        self.api_key = CONFIG.finnhub_api_key

    def get_quote(self, ticker: str) -> tuple[dict, SourceRecord]:
        params = {"symbol": ticker, "token": self.api_key}
        payload, fetched_at, cache_hit = self.cached_get_json(
            "quote", f"{_BASE_URL}/quote", params=params, ttl_seconds=60
        )
        if payload.get("t", 0) == 0:
            raise ValueError(f"Finnhub returned empty quote for {ticker!r}: {payload}")
        obs_time = datetime.fromtimestamp(payload["t"], tz=timezone.utc)
        record = self.make_record(
            "quote", fields=payload, fetched_at=fetched_at, observation_time=obs_time,
            entity=ticker, license_note=LICENSE_NOTE, cache_hit=cache_hit,
        )
        return payload, record

    def get_company_news(self, ticker: str, from_date: date, to_date: date) -> tuple[list[dict], SourceRecord]:
        """Returns raw articles: headline, summary, source, url, datetime (unix), category.
        Timestamps preserved verbatim for downstream leakage auditing (never allow an
        article's timestamp to be treated as earlier than its actual publish time)."""
        params = {
            "symbol": ticker,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "token": self.api_key,
        }
        payload, fetched_at, cache_hit = self.cached_get_json(
            "company_news", f"{_BASE_URL}/company-news", params=params, ttl_seconds=1800
        )
        if not isinstance(payload, list):
            raise ValueError(f"Finnhub company-news returned unexpected payload for {ticker!r}: {payload}")
        obs_time = None
        if payload:
            obs_time = datetime.fromtimestamp(max(a.get("datetime", 0) for a in payload), tz=timezone.utc)
        record = self.make_record(
            "company_news",
            fields={"n_articles": len(payload), "from": str(from_date), "to": str(to_date)},
            fetched_at=fetched_at, observation_time=obs_time, entity=ticker,
            license_note=LICENSE_NOTE, cache_hit=cache_hit,
        )
        return payload, record
