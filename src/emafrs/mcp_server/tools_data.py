"""Plain, MCP-independent functions wrapping the DAL clients for the helper dev server.

Kept separate from server.py so they can be unit-tested directly (call the function, get a
dict back) without going through the MCP stdio transport. Every function returns
{"ok": bool, ...} -- on failure (missing key, provider unreachable, no cache) it returns
{"ok": False, "error": "..."} rather than raising through the MCP boundary, and it never
fabricates a substitute value for a source that failed.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from emafrs.data.config import CONFIG
from emafrs.data.schema import SourceRecord


def _df_to_records(df: pd.DataFrame, max_rows: int = 60) -> list[dict]:
    out = df.tail(max_rows).reset_index()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].astype(str)
    return out.to_dict(orient="records")


def _record_meta(record: SourceRecord) -> dict[str, Any]:
    return {
        "provider": record.provider,
        "endpoint": record.endpoint,
        "fetched_at": record.fetched_at.isoformat(),
        "observation_time": record.observation_time.isoformat() if record.observation_time else None,
        "entity": record.entity,
        "license_note": record.license_note,
        "cache_hit": record.cache_hit,
    }


def _err(exc: Exception) -> dict[str, Any]:
    return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def get_ohlcv(ticker: str, period: str = "6mo", interval: str = "1d") -> dict:
    """Daily/intraday OHLCV history for a ticker via yfinance (no key required)."""
    try:
        from emafrs.data.yfinance_client import YFinanceClient

        df, record = YFinanceClient().get_ohlcv(ticker, period=period, interval=interval)
        return {"ok": True, "rows": _df_to_records(df), "meta": _record_meta(record)}
    except Exception as exc:
        return _err(exc)


def get_alphavantage_daily(ticker: str) -> dict:
    """Daily adjusted OHLCV from Alpha Vantage, used as a validation source against yfinance."""
    if not CONFIG.is_configured("alphavantage"):
        return {"ok": False, "error": "ALPHAVANTAGE_API_KEY not configured"}
    try:
        from emafrs.data.alphavantage_client import AlphaVantageClient

        df, record = AlphaVantageClient().get_daily_adjusted(ticker)
        return {"ok": True, "rows": _df_to_records(df), "meta": _record_meta(record)}
    except Exception as exc:
        return _err(exc)


def get_quote(ticker: str) -> dict:
    """Latest quote snapshot from Finnhub."""
    if not CONFIG.is_configured("finnhub"):
        return {"ok": False, "error": "FINNHUB_API_KEY not configured"}
    try:
        from emafrs.data.finnhub_client import FinnhubClient

        payload, record = FinnhubClient().get_quote(ticker)
        return {"ok": True, "quote": payload, "meta": _record_meta(record)}
    except Exception as exc:
        return _err(exc)


def get_company_news(ticker: str, days_back: int = 7) -> dict:
    """Company news headlines from Finnhub for the trailing `days_back` days."""
    if not CONFIG.is_configured("finnhub"):
        return {"ok": False, "error": "FINNHUB_API_KEY not configured"}
    try:
        from emafrs.data.finnhub_client import FinnhubClient

        today = date.today()
        articles, record = FinnhubClient().get_company_news(ticker, today - timedelta(days=days_back), today)
        return {"ok": True, "articles": articles[:30], "n_total": len(articles), "meta": _record_meta(record)}
    except Exception as exc:
        return _err(exc)


def search_news(query: str, days_back: int = 7) -> dict:
    """General news search via NewsAPI.org for the trailing `days_back` days (free tier: ~1 month max lookback)."""
    if not CONFIG.is_configured("newsapi"):
        return {"ok": False, "error": "NEWS_API_KEY not configured"}
    try:
        from emafrs.data.newsapi_client import NewsAPIClient

        today = date.today()
        articles, record = NewsAPIClient().search(query, today - timedelta(days=days_back), today)
        return {"ok": True, "articles": articles[:30], "n_total": len(articles), "meta": _record_meta(record)}
    except Exception as exc:
        return _err(exc)


def search_youtube_videos(query: str, days_back: int = 30, max_results: int = 5) -> dict:
    """Search for relevant financial-communication videos. Costs 100 YouTube quota units
    per call (10,000/day budget) -- use sparingly, prefer a specific query over broad terms."""
    if not CONFIG.is_configured("youtube"):
        return {"ok": False, "error": "YOUTUBE_API_KEY not configured"}
    try:
        from emafrs.data.youtube_client import YouTubeClient

        now = datetime.now(timezone.utc)
        items, record = YouTubeClient().search_videos(
            query, now - timedelta(days=days_back), now, max_results=max_results
        )
        return {"ok": True, "videos": items, "meta": _record_meta(record)}
    except Exception as exc:
        return _err(exc)


def get_youtube_transcript(video_id: str) -> dict:
    """Fetch transcript text for a YouTube video id, if available. This is a TEXT modality
    only -- no audio/video signal is extracted here."""
    if not CONFIG.is_configured("youtube"):
        return {"ok": False, "error": "YOUTUBE_API_KEY not configured"}
    try:
        from emafrs.data.youtube_client import YouTubeClient

        text, record = YouTubeClient().get_transcript_text(video_id)
        return {"ok": True, "transcript_available": text is not None, "transcript": text, "meta": _record_meta(record)}
    except Exception as exc:
        return _err(exc)


def get_fred_series(series_id: str, days_back: int = 730) -> dict:
    """Macroeconomic series from FRED. Pass a raw FRED series id (e.g. 'DFF', 'CPIAUCSL',
    'UNRATE') or one of the catalog keys in emafrs.data.fred_client.SERIES_CATALOG."""
    if not CONFIG.is_configured("fred"):
        return {"ok": False, "error": "FRED_API_KEY not configured"}
    try:
        from emafrs.data.fred_client import FREDClient, SERIES_CATALOG

        resolved_id = SERIES_CATALOG.get(series_id, series_id)
        start = date.today() - timedelta(days=days_back)
        df, record = FREDClient().get_series(resolved_id, observation_start=start)
        return {"ok": True, "series": _df_to_records(df), "resolved_series_id": resolved_id, "meta": _record_meta(record)}
    except Exception as exc:
        return _err(exc)


def sec_ticker_to_cik(ticker: str) -> dict:
    """Resolve a ticker to its SEC EDGAR CIK number."""
    try:
        from emafrs.data.sec_edgar_client import SECEdgarClient

        cik, record = SECEdgarClient().ticker_to_cik(ticker)
        return {"ok": cik is not None, "cik": cik, "meta": _record_meta(record)}
    except Exception as exc:
        return _err(exc)


def sec_full_text_search(query: str, forms: str | None = None, days_back: int = 90) -> dict:
    """Full-text search across SEC EDGAR filings since 2001; restrict to a form type
    (e.g. '8-K') for material-event disclosures relevant to event/hype detection."""
    try:
        from emafrs.data.sec_edgar_client import SECEdgarClient

        start = date.today() - timedelta(days=days_back)
        results, record = SECEdgarClient().full_text_search(query, forms=forms, start=start)
        return {"ok": True, "results": results, "meta": _record_meta(record)}
    except Exception as exc:
        return _err(exc)


def check_data_source_health(live: bool = False) -> dict:
    """Report configuration + (optionally) live reachability of every DAL source. live=True
    performs one real request per source -- costs real quota (notably YouTube: 100 units)."""
    from emafrs.data.health import check_all

    health = check_all(live=live)
    return {
        "ok": True,
        "sources": {
            name: {
                "configured": h.configured,
                "reachable": h.reachable,
                "last_success_at": h.last_success_at.isoformat() if h.last_success_at else None,
                "last_error": h.last_error,
                "notes": h.notes,
            }
            for name, h in health.items()
        },
    }
