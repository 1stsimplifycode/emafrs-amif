"""NewsAPI.org client: general market/company news.

Free ("Developer") tier constraints, per newsapi.org/pricing as of this writing:
100 requests/day, articles no older than ~1 month, non-commercial use only, requires
attribution, and (important for leakage control) the free tier serves articles with
some ingestion delay -- treat `publishedAt` as authoritative, never assume same-day
availability.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from emafrs.data.base import BaseAPIClient
from emafrs.data.config import CONFIG
from emafrs.data.schema import SourceRecord

_BASE_URL = "https://newsapi.org/v2/everything"
LICENSE_NOTE = (
    "NewsAPI.org free Developer tier: 100 requests/day, ~1 month lookback, "
    "non-commercial use only, attribution required. See newsapi.org/terms."
)

# Real domains NewsAPI indexes for these publishers -- used with the `domains` filter
# above so a "CNBC/Bloomberg" query is actually restricted to their real published
# articles, not a generic keyword search that might return any outlet.
KNOWN_FINANCIAL_NEWS_DOMAINS = "cnbc.com,bloomberg.com"


class NewsAPIClient(BaseAPIClient):
    provider = "newsapi"

    def __init__(self):
        if not CONFIG.news_api_key:
            raise RuntimeError("NEWS_API_KEY not configured")
        self.api_key = CONFIG.news_api_key

    def search(
        self,
        query: str,
        from_date: date,
        to_date: date,
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 100,
        domains: str | None = None,
    ) -> tuple[list[dict], SourceRecord]:
        """Raises ValueError with the provider's own error message on failure/quota
        exhaustion rather than returning an empty list silently, so callers can
        distinguish 'no news found' from 'source unavailable'.

        `domains`: NewsAPI's real `domains` filter (comma-separated, e.g.
        "cnbc.com,bloomberg.com") -- restricts results to those publishers specifically,
        rather than a generic keyword search across every source NewsAPI indexes. This
        parameter existed on NewsAPI's real API all along; this client just didn't expose
        it until now."""
        params = {
            "q": query,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "language": language,
            "sortBy": sort_by,
            "pageSize": page_size,
            "apiKey": self.api_key,
        }
        if domains:
            params["domains"] = domains
        payload, fetched_at, cache_hit = self.cached_get_json(
            "everything", _BASE_URL, params=params, ttl_seconds=1800
        )
        if payload.get("status") != "ok":
            raise ValueError(f"NewsAPI error for query={query!r}: {payload.get('message', payload)}")
        articles = payload.get("articles", [])
        obs_time = None
        parsed_times = []
        for a in articles:
            try:
                parsed_times.append(datetime.fromisoformat(a["publishedAt"].replace("Z", "+00:00")))
            except (KeyError, ValueError):
                continue
        if parsed_times:
            obs_time = max(parsed_times)

        record = self.make_record(
            "everything",
            fields={"query": query, "n_articles": len(articles), "total_results": payload.get("totalResults")},
            fetched_at=fetched_at, observation_time=obs_time, entity=query,
            license_note=LICENSE_NOTE, cache_hit=cache_hit,
        )
        return articles, record
