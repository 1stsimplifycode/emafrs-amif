"""Simple TTL disk cache for raw provider responses.

Purpose: respect free-tier rate limits (NewsAPI: 100 req/day; Alpha Vantage: 25 req/day,
5 req/min; YouTube: 10,000 quota units/day) without ever presenting cached data as live.
Every read reports its own age; `SourceRecord.cache_hit` and `fetched_at` are set from the
cache entry's original fetch time, not the read time.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from emafrs.data.config import CONFIG


def _cache_key(provider: str, endpoint: str, params: dict[str, Any]) -> str:
    raw = json.dumps({"provider": provider, "endpoint": endpoint, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


@dataclass
class CacheEntry:
    payload: Any
    fetched_at: datetime
    cache_hit: bool


class DiskCache:
    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or CONFIG.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, provider: str, endpoint: str, params: dict[str, Any]) -> Path:
        key = _cache_key(provider, endpoint, params)
        provider_dir = self.cache_dir / provider
        provider_dir.mkdir(parents=True, exist_ok=True)
        return provider_dir / f"{endpoint}_{key}.json"

    def get(self, provider: str, endpoint: str, params: dict[str, Any], ttl_seconds: float) -> Optional[CacheEntry]:
        path = self._path(provider, endpoint, params)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                blob = json.load(f)
            fetched_at = datetime.fromisoformat(blob["fetched_at"])
            age = (datetime.now(timezone.utc) - fetched_at).total_seconds()
            if age > ttl_seconds:
                return None
            return CacheEntry(payload=blob["payload"], fetched_at=fetched_at, cache_hit=True)
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            return None

    def get_stale(self, provider: str, endpoint: str, params: dict[str, Any]) -> Optional[CacheEntry]:
        """Return a cache entry regardless of TTL, for fallback-when-rate-limited use only.
        Caller MUST mark the resulting data as stale to downstream consumers."""
        path = self._path(provider, endpoint, params)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                blob = json.load(f)
            return CacheEntry(
                payload=blob["payload"],
                fetched_at=datetime.fromisoformat(blob["fetched_at"]),
                cache_hit=True,
            )
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            return None

    def set(self, provider: str, endpoint: str, params: dict[str, Any], payload: Any) -> datetime:
        path = self._path(provider, endpoint, params)
        fetched_at = datetime.now(timezone.utc)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"fetched_at": fetched_at.isoformat(), "payload": payload}, f, default=str)
        return fetched_at


CACHE = DiskCache()
