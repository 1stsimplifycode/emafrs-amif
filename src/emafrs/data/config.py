"""Loads API configuration from environment (.env) and reports which sources are usable.

Rule enforced here: a source is only ever used if its key is actually present. Nothing
downstream should fall back to fabricated/mocked data when a key is missing — callers
must check `is_configured()` and report absence explicitly (see docs/DATA_SOURCES_AND_APIS.md).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)
else:
    load_dotenv()  # fall back to process env / any .env on default search path


@dataclass(frozen=True)
class DataSourceConfig:
    youtube_api_key: str | None = os.getenv("YOUTUBE_API_KEY") or None
    news_api_key: str | None = os.getenv("NEWS_API_KEY") or None
    alphavantage_api_key: str | None = os.getenv("ALPHAVANTAGE_API_KEY") or None
    finnhub_api_key: str | None = os.getenv("FINNHUB_API_KEY") or None
    fred_api_key: str | None = os.getenv("FRED_API_KEY") or None
    sec_edgar_user_agent: str | None = (
        os.getenv("SEC_EDGAR_USER_AGENT") or "EMAFRS-Research-Reconstruction (contact not set)"
    )
    cache_dir: Path = _PROJECT_ROOT / "data" / "cache"

    def is_configured(self, provider: str) -> bool:
        return {
            "youtube": bool(self.youtube_api_key),
            "newsapi": bool(self.news_api_key),
            "alphavantage": bool(self.alphavantage_api_key),
            "finnhub": bool(self.finnhub_api_key),
            "fred": bool(self.fred_api_key),
            "yfinance": True,  # no key required
            "sec_edgar": True,  # no key required, only a User-Agent
        }.get(provider, False)

    def configured_sources(self) -> dict[str, bool]:
        return {p: self.is_configured(p) for p in
                ("yfinance", "alphavantage", "finnhub", "newsapi", "youtube", "fred", "sec_edgar")}


CONFIG = DataSourceConfig()
CONFIG.cache_dir.mkdir(parents=True, exist_ok=True)
