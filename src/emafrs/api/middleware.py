"""Logging, security headers, and rate-limiting middleware.

Rate limiting is a simple in-memory fixed-window counter per client IP -- correct for a
single-process deployment only. A production multi-worker deployment needs a shared store
(Redis) instead; noted here and in docs/PRODUCTION_DEPLOYMENT.md rather than pretended away.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("emafrs.api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs method, path, status, and latency only -- never request/response bodies or
    headers (which could carry an Authorization/API-key value) or raw query strings."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, elapsed_ms)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response


class InMemoryRateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window rate limiter, default 60 requests/minute per client IP."""

    def __init__(self, app, requests_per_window: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self._counts: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - self.window_seconds

        timestamps = self._counts[client_ip]
        timestamps[:] = [t for t in timestamps if t > window_start]

        if len(timestamps) >= self.requests_per_window:
            return Response(content="Rate limit exceeded", status_code=429)

        timestamps.append(now)
        return await call_next(request)
