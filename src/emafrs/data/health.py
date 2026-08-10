"""Aggregate health/freshness check across all configured data sources.

Feeds the dashboard's "API health and data freshness" panel (build brief Section 29).
Never reports a source as healthy without an actual probe; unconfigured sources are
reported as such rather than omitted, so the dashboard shows the true coverage picture.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

from emafrs.data.config import CONFIG
from emafrs.data.schema import SourceHealth

logger = logging.getLogger("emafrs.data.health")

_PROBE_TICKER = "AAPL"


def _probe_yfinance() -> SourceHealth:
    from emafrs.data.yfinance_client import YFinanceClient

    h = SourceHealth(provider="yfinance", configured=True)
    try:
        _, record = YFinanceClient().get_ohlcv(_PROBE_TICKER, period="5d", interval="1d")
        h.reachable = True
        h.last_success_at = record.fetched_at
    except Exception as exc:
        h.reachable = False
        h.last_error = str(exc)
    return h


def _probe_alphavantage() -> SourceHealth:
    h = SourceHealth(provider="alphavantage", configured=CONFIG.is_configured("alphavantage"))
    if not h.configured:
        h.notes.append("ALPHAVANTAGE_API_KEY not set")
        return h
    try:
        from emafrs.data.alphavantage_client import AlphaVantageClient

        _, record = AlphaVantageClient().get_daily_adjusted(_PROBE_TICKER, outputsize="compact")
        h.reachable = True
        h.last_success_at = record.fetched_at
    except Exception as exc:
        h.reachable = False
        h.last_error = str(exc)
    return h


def _probe_finnhub() -> SourceHealth:
    h = SourceHealth(provider="finnhub", configured=CONFIG.is_configured("finnhub"))
    if not h.configured:
        h.notes.append("FINNHUB_API_KEY not set")
        return h
    try:
        from emafrs.data.finnhub_client import FinnhubClient

        _, record = FinnhubClient().get_quote(_PROBE_TICKER)
        h.reachable = True
        h.last_success_at = record.fetched_at
    except Exception as exc:
        h.reachable = False
        h.last_error = str(exc)
    return h


def _probe_newsapi() -> SourceHealth:
    h = SourceHealth(provider="newsapi", configured=CONFIG.is_configured("newsapi"))
    if not h.configured:
        h.notes.append("NEWS_API_KEY not set")
        return h
    try:
        from emafrs.data.newsapi_client import NewsAPIClient

        today = date.today()
        _, record = NewsAPIClient().search(_PROBE_TICKER, today - timedelta(days=2), today, page_size=1)
        h.reachable = True
        h.last_success_at = record.fetched_at
    except Exception as exc:
        h.reachable = False
        h.last_error = str(exc)
    return h


def _probe_youtube() -> SourceHealth:
    h = SourceHealth(provider="youtube", configured=CONFIG.is_configured("youtube"))
    if not h.configured:
        h.notes.append("YOUTUBE_API_KEY not set")
        return h
    try:
        from emafrs.data.youtube_client import YouTubeClient

        now = datetime.now(timezone.utc)
        _, record = YouTubeClient().search_videos(
            "federal reserve", now - timedelta(days=7), now, max_results=1
        )
        h.reachable = True
        h.last_success_at = record.fetched_at
        h.notes.append("search.list costs 100 quota units -- probe sparingly")
    except Exception as exc:
        h.reachable = False
        h.last_error = str(exc)
    return h


def _probe_fred() -> SourceHealth:
    h = SourceHealth(provider="fred", configured=CONFIG.is_configured("fred"))
    if not h.configured:
        h.notes.append("FRED_API_KEY not set")
        return h
    try:
        from emafrs.data.fred_client import FREDClient, SERIES_CATALOG

        _, record = FREDClient().get_series(SERIES_CATALOG["fed_funds_rate"])
        h.reachable = True
        h.last_success_at = record.fetched_at
    except Exception as exc:
        h.reachable = False
        h.last_error = str(exc)
    return h


def _probe_sec_edgar() -> SourceHealth:
    h = SourceHealth(provider="sec_edgar", configured=True)
    try:
        from emafrs.data.sec_edgar_client import SECEdgarClient

        _, record = SECEdgarClient().ticker_to_cik(_PROBE_TICKER)
        h.reachable = True
        h.last_success_at = record.fetched_at
    except Exception as exc:
        h.reachable = False
        h.last_error = str(exc)
    return h


_PROBES = {
    "yfinance": _probe_yfinance,
    "alphavantage": _probe_alphavantage,
    "finnhub": _probe_finnhub,
    "newsapi": _probe_newsapi,
    "youtube": _probe_youtube,
    "fred": _probe_fred,
    "sec_edgar": _probe_sec_edgar,
}


def check_all(live: bool = False) -> dict[str, SourceHealth]:
    """If live=False (default), only reports configuration status without making network
    calls -- use live=True sparingly (e.g. a scheduled job), since several probes consume
    real quota (notably YouTube search.list)."""
    if not live:
        return {
            name: SourceHealth(provider=name, configured=CONFIG.is_configured(name))
            for name in _PROBES
        }
    return {name: probe() for name, probe in _PROBES.items()}
