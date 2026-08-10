"""Shared HTTP plumbing: retry/backoff, timeout, and cache-through GET for all DAL clients."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from emafrs.data.cache import CACHE
from emafrs.data.schema import SourceRecord

logger = logging.getLogger("emafrs.data")

_DEFAULT_TIMEOUT = 15  # seconds


class RateLimitedError(RuntimeError):
    """Raised when a provider signals rate-limit exhaustion (HTTP 429 or provider-specific body)."""


class ProviderUnavailableError(RuntimeError):
    """Raised after retries are exhausted and no usable cache fallback exists."""


class RetryableHTTPError(RuntimeError):
    pass


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=1, max=20),
    retry=retry_if_exception_type((RetryableHTTPError, requests.Timeout, requests.ConnectionError)),
)
def _get_with_retry(url: str, params: dict[str, Any] | None, headers: dict[str, str] | None) -> requests.Response:
    resp = requests.get(url, params=params, headers=headers, timeout=_DEFAULT_TIMEOUT)
    if resp.status_code == 429:
        raise RateLimitedError(f"Rate limited by {url}")
    if resp.status_code >= 500:
        raise RetryableHTTPError(f"{resp.status_code} from {url}")
    return resp


class BaseAPIClient:
    """Cache-through GET with retry/backoff and a stale-cache fallback on rate-limit/outage."""

    provider: str = "base"

    def cached_get_json(
        self,
        endpoint: str,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        ttl_seconds: float = 900,
    ) -> tuple[Any, datetime, bool]:
        """Returns (payload, fetched_at, cache_hit). Raises ProviderUnavailableError if the
        live call fails AND no cache (even stale) exists — callers must not fabricate data."""
        params = params or {}
        cached = CACHE.get(self.provider, endpoint, params, ttl_seconds)
        if cached is not None:
            return cached.payload, cached.fetched_at, True

        try:
            resp = _get_with_retry(url, params, headers)
        except (RateLimitedError, requests.RequestException, RetryableHTTPError) as exc:
            logger.warning("%s live fetch failed (%s); trying stale cache fallback", self.provider, exc)
            stale = CACHE.get_stale(self.provider, endpoint, params)
            if stale is not None:
                return stale.payload, stale.fetched_at, True
            raise ProviderUnavailableError(f"{self.provider}:{endpoint} unavailable and no cache exists") from exc

        if resp.status_code == 429:
            stale = CACHE.get_stale(self.provider, endpoint, params)
            if stale is not None:
                return stale.payload, stale.fetched_at, True
            raise ProviderUnavailableError(f"{self.provider}:{endpoint} rate-limited and no cache exists")

        if not resp.ok:
            stale = CACHE.get_stale(self.provider, endpoint, params)
            if stale is not None:
                logger.warning("%s returned %s; serving stale cache", self.provider, resp.status_code)
                return stale.payload, stale.fetched_at, True
            raise ProviderUnavailableError(f"{self.provider}:{endpoint} HTTP {resp.status_code}: {resp.text[:300]}")

        payload = resp.json()
        fetched_at = CACHE.set(self.provider, endpoint, params, payload)
        return payload, fetched_at, False

    def make_record(
        self,
        endpoint: str,
        fields: dict[str, Any],
        fetched_at: Optional[datetime] = None,
        observation_time: Optional[datetime] = None,
        entity: Optional[str] = None,
        license_note: str = "",
        cache_hit: bool = False,
    ) -> SourceRecord:
        return SourceRecord(
            provider=self.provider,
            endpoint=endpoint,
            fetched_at=fetched_at or datetime.now(timezone.utc),
            observation_time=observation_time,
            entity=entity,
            fields=fields,
            license_note=license_note,
            cache_hit=cache_hit,
        )
