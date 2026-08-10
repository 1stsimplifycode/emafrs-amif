"""FRED (Federal Reserve Economic Data) client: macroeconomic series.

Free, no meaningful rate limit for research use (published guidance: 120 requests/minute).
Used for macro context features (e.g. effective fed funds rate, CPI, unemployment) that
condition regime interpretation but are not claimed as microstructure or affective signals.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from emafrs.data.base import BaseAPIClient
from emafrs.data.config import CONFIG
from emafrs.data.schema import SourceRecord

_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
LICENSE_NOTE = "FRED public domain data (U.S. government work). See fred.stlouisfed.org/legal/."

# A small, documented set of macro series relevant to regime/context features.
SERIES_CATALOG = {
    "fed_funds_rate": "DFF",
    "cpi_all_urban": "CPIAUCSL",
    "unemployment_rate": "UNRATE",
    "vix_proxy_10y_yield": "DGS10",
    "m2_money_supply": "M2SL",
}


class FREDClient(BaseAPIClient):
    provider = "fred"

    def __init__(self):
        if not CONFIG.fred_api_key:
            raise RuntimeError("FRED_API_KEY not configured")
        self.api_key = CONFIG.fred_api_key

    def get_series(
        self, series_id: str, observation_start: date | None = None, observation_end: date | None = None
    ) -> tuple[pd.DataFrame, SourceRecord]:
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        if observation_start:
            params["observation_start"] = observation_start.isoformat()
        if observation_end:
            params["observation_end"] = observation_end.isoformat()

        payload, fetched_at, cache_hit = self.cached_get_json(
            f"series_{series_id}", _BASE_URL, params=params, ttl_seconds=24 * 3600
        )
        obs = payload.get("observations")
        if not obs:
            raise ValueError(f"FRED returned no observations for series_id={series_id!r}: {payload}")
        df = pd.DataFrame(obs)[["date", "value"]]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED uses "." for missing
        df = df.set_index("date").dropna()

        record = self.make_record(
            f"series_{series_id}",
            fields={"series_id": series_id, "n_obs": len(df)},
            fetched_at=fetched_at,
            observation_time=df.index.max().to_pydatetime().replace(tzinfo=timezone.utc) if len(df) else None,
            entity=series_id,
            license_note=LICENSE_NOTE,
            cache_hit=cache_hit,
        )
        return df, record
