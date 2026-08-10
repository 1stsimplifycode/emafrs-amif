"""Alpha Vantage client: supplementary/validation market data source.

Used to cross-check yfinance rather than blindly merged with it -- discrepancies are
surfaced via provenance tags on both records rather than silently reconciled (build brief
Section 10: "Do not blindly merge conflicting sources. Implement source provenance.").

Free tier (as published at alphavantage.co/premium as of this writing; verify current
limits before relying on them): 25 requests/day, 5 requests/minute.

REAL COMPATIBILITY ISSUE FOUND DURING USE (not hypothetical): `TIME_SERIES_DAILY_ADJUSTED`
now returns "This is a premium endpoint" on the free tier -- Alpha Vantage moved it behind
a paywall at some point after this client was first written. `get_daily()` below uses the
still-free `TIME_SERIES_DAILY` endpoint instead, which does NOT include a dividend-adjusted
close column. Documented here and in docs/DATA_SOURCES_AND_APIS.md rather than silently
patched over.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from emafrs.data.base import BaseAPIClient
from emafrs.data.config import CONFIG
from emafrs.data.schema import SourceRecord

_BASE_URL = "https://www.alphavantage.co/query"
LICENSE_NOTE = (
    "Alpha Vantage free tier: 25 requests/day, 5/minute (subject to change by provider). "
    "Redistribution restrictions apply per alphavantage.co/terms_of_service."
)


class AlphaVantageClient(BaseAPIClient):
    provider = "alphavantage"

    def __init__(self):
        if not CONFIG.alphavantage_api_key:
            raise RuntimeError("ALPHAVANTAGE_API_KEY not configured")
        self.api_key = CONFIG.alphavantage_api_key

    def get_daily(self, ticker: str, outputsize: str = "compact") -> tuple[pd.DataFrame, SourceRecord]:
        """Daily OHLCV, NOT dividend/split-adjusted (TIME_SERIES_DAILY_ADJUSTED is a
        premium-only endpoint on the current free tier -- see module docstring)."""
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": outputsize,
            "apikey": self.api_key,
        }
        payload, fetched_at, cache_hit = self.cached_get_json(
            "daily", _BASE_URL, params=params, ttl_seconds=6 * 3600
        )
        series = payload.get("Time Series (Daily)")
        if not series:
            raise ValueError(
                f"Alpha Vantage returned no daily series for {ticker!r}: {payload.get('Note') or payload.get('Information') or payload}"
            )
        df = pd.DataFrame.from_dict(series, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.columns = ["open", "high", "low", "close", "volume"]
        df = df.astype(float)

        record = self.make_record(
            "daily_adjusted",
            fields={"n_rows": len(df), "outputsize": outputsize},
            fetched_at=fetched_at,
            observation_time=df.index.max().to_pydatetime().replace(tzinfo=timezone.utc),
            entity=ticker,
            license_note=LICENSE_NOTE,
            cache_hit=cache_hit,
        )
        return df, record

    def get_technical_indicator(
        self, ticker: str, function: str, interval: str = "daily", time_period: int = 14, series_type: str = "close"
    ) -> tuple[pd.DataFrame, SourceRecord]:
        """function e.g. 'RSI', 'MACD', 'SMA', 'EMA', 'BBANDS' -- used only as a
        cross-check against locally computed indicators (src/emafrs/features/technical.py),
        not as the primary indicator source."""
        params = {
            "function": function,
            "symbol": ticker,
            "interval": interval,
            "time_period": time_period,
            "series_type": series_type,
            "apikey": self.api_key,
        }
        payload, fetched_at, cache_hit = self.cached_get_json(
            f"indicator_{function}", _BASE_URL, params=params, ttl_seconds=6 * 3600
        )
        key = next((k for k in payload if k.startswith("Technical Analysis")), None)
        if key is None:
            raise ValueError(f"Alpha Vantage returned no indicator series for {ticker!r}/{function}: {payload}")
        df = pd.DataFrame.from_dict(payload[key], orient="index").astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        record = self.make_record(
            f"indicator_{function}",
            fields={"n_rows": len(df), "function": function, "interval": interval},
            fetched_at=fetched_at,
            observation_time=df.index.max().to_pydatetime().replace(tzinfo=timezone.utc),
            entity=ticker,
            license_note=LICENSE_NOTE,
            cache_hit=cache_hit,
        )
        return df, record

    def get_intraday(self, ticker: str, interval: str = "5min") -> tuple[pd.DataFrame, SourceRecord]:
        """Used as a partial proxy input for MMIE microstructure signals (bar-level, not
        tick-level -- see docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md row #4)."""
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": ticker,
            "interval": interval,
            "outputsize": "full",
            "apikey": self.api_key,
        }
        payload, fetched_at, cache_hit = self.cached_get_json(
            f"intraday_{interval}", _BASE_URL, params=params, ttl_seconds=900
        )
        key = f"Time Series ({interval})"
        series = payload.get(key)
        if not series:
            raise ValueError(f"Alpha Vantage returned no intraday series for {ticker!r}: {payload}")
        df = pd.DataFrame.from_dict(series, orient="index")
        df.columns = ["open", "high", "low", "close", "volume"]
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        record = self.make_record(
            f"intraday_{interval}",
            fields={"n_rows": len(df), "interval": interval},
            fetched_at=fetched_at,
            observation_time=df.index.max().to_pydatetime().replace(tzinfo=timezone.utc),
            entity=ticker,
            license_note=LICENSE_NOTE,
            cache_hit=cache_hit,
        )
        return df, record
