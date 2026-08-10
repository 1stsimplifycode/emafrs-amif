"""Unit tests for the DAL disk cache: no network access."""
import tempfile
from pathlib import Path

from emafrs.data.cache import DiskCache


def test_cache_roundtrip_within_ttl():
    with tempfile.TemporaryDirectory() as tmp:
        cache = DiskCache(cache_dir=Path(tmp))
        cache.set("providerX", "endpointY", {"a": 1}, {"hello": "world"})
        entry = cache.get("providerX", "endpointY", {"a": 1}, ttl_seconds=3600)
        assert entry is not None
        assert entry.payload == {"hello": "world"}
        assert entry.cache_hit is True


def test_cache_miss_when_ttl_expired():
    with tempfile.TemporaryDirectory() as tmp:
        cache = DiskCache(cache_dir=Path(tmp))
        cache.set("providerX", "endpointY", {"a": 1}, {"hello": "world"})
        entry = cache.get("providerX", "endpointY", {"a": 1}, ttl_seconds=-1)
        assert entry is None


def test_cache_miss_for_unseen_key():
    with tempfile.TemporaryDirectory() as tmp:
        cache = DiskCache(cache_dir=Path(tmp))
        entry = cache.get("providerX", "endpointY", {"never": "set"}, ttl_seconds=3600)
        assert entry is None


def test_stale_fallback_ignores_ttl():
    with tempfile.TemporaryDirectory() as tmp:
        cache = DiskCache(cache_dir=Path(tmp))
        cache.set("providerX", "endpointY", {"a": 1}, {"hello": "world"})
        # even with an already-expired TTL, get_stale must still return it
        entry = cache.get("providerX", "endpointY", {"a": 1}, ttl_seconds=-1)
        assert entry is None
        stale = cache.get_stale("providerX", "endpointY", {"a": 1})
        assert stale is not None
        assert stale.payload == {"hello": "world"}


def test_different_params_are_different_cache_entries():
    with tempfile.TemporaryDirectory() as tmp:
        cache = DiskCache(cache_dir=Path(tmp))
        cache.set("p", "e", {"ticker": "AAPL"}, {"v": 1})
        cache.set("p", "e", {"ticker": "MSFT"}, {"v": 2})
        a = cache.get("p", "e", {"ticker": "AAPL"}, ttl_seconds=3600)
        m = cache.get("p", "e", {"ticker": "MSFT"}, ttl_seconds=3600)
        assert a.payload == {"v": 1}
        assert m.payload == {"v": 2}
