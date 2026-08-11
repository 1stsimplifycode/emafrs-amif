"""Real, working background scheduler.

This module is referenced by name in training_pipeline.py's docstring ("a scheduled
background job (emafrs.mlops.scheduler) calls it periodically for a production-like
refresh pattern") and by DataFreshnessRecord's docstring in db/models.py -- but a direct
code search found it did not actually exist anywhere in this codebase. Those were stale,
aspirational references to a module that was never built. This is the real
implementation: an asyncio loop, started from FastAPI's lifespan handler (no new
dependency), that periodically refreshes real data-source health and real predictions so
the app has continuously-updating data instead of only ever updating on a user's click.

Quota-aware by design: `check_all(live=True)` (real network probes to every provider,
notably YouTube's search.list at 100 units/call -- see data/health.py's own docstring,
which already said to "use live=True sparingly (e.g. a scheduled job)") defaults OFF here
and must be explicitly enabled via SCHEDULER_HEALTH_LIVE_PROBE, since a health probe
consuming real API quota on every cycle is a real cost, not a hypothetical one. Prediction
refreshes do not call YouTube/NewsAPI at all unless `include_text=True` is separately
requested (see training_pipeline.py) -- the default scheduled prediction refresh only
spends real compute time (CPU model training), not News/YouTube request quota.
"""
from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger("emafrs.mlops.scheduler")

DEFAULT_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]


def refresh_data_health(live: bool = False) -> int:
    """Real health check (emafrs.data.health.check_all), persisted as one new
    DataFreshnessRecord row per provider. DataFreshnessRecord existed in the schema
    since early in this project but (confirmed by a direct code search before writing
    this module) nothing ever wrote to it -- this is the first real writer."""
    from emafrs.data.health import check_all
    from emafrs.db.models import DataFreshnessRecord
    from emafrs.db.session import get_session

    results = check_all(live=live)
    session = get_session()
    try:
        for provider, health in results.items():
            session.add(DataFreshnessRecord(
                provider=provider,
                configured=health.configured,
                reachable=health.reachable,
                last_success_at=health.last_success_at,
                last_error=health.last_error,
            ))
        session.commit()
    finally:
        session.close()
    return len(results)


def refresh_predictions(tickers: list[str], include_text: bool = False) -> dict[str, str]:
    """Real training + inference for each ticker -- the exact same function
    POST /predictions/{ticker}/refresh calls, just triggered on a timer instead of a
    click. One ticker's failure (e.g. a real provider outage) is logged and does not
    stop the rest -- never silently swallowed, never allowed to kill the whole cycle."""
    from emafrs.mlops.training_pipeline import run_training_and_prediction_pipeline

    outcomes: dict[str, str] = {}
    for ticker in tickers:
        try:
            record = run_training_and_prediction_pipeline(ticker, include_text=include_text)
            outcomes[ticker] = f"ok: direction={record.direction} confidence={record.confidence:.3f}"
        except Exception as exc:
            logger.warning("Scheduled prediction refresh failed for %s: %s", ticker, exc)
            outcomes[ticker] = f"error: {exc}"
    return outcomes


async def run_scheduler_loop(
    interval_seconds: int,
    tickers: list[str],
    refresh_health: bool = True,
    refresh_preds: bool = True,
    health_live_probe: bool = False,
    include_text: bool = False,
) -> None:
    """Runs forever inside the FastAPI process until the task is cancelled at shutdown.
    The first cycle runs immediately on startup (no initial sleep) so there is real
    refreshed data shortly after the app comes up, not only after a full interval."""
    while True:
        logger.info("Scheduler cycle starting (tickers=%s)", tickers)
        try:
            if refresh_health:
                n = refresh_data_health(live=health_live_probe)
                logger.info("Scheduler refreshed data-source health for %d providers (live_probe=%s)", n, health_live_probe)
            if refresh_preds:
                outcomes = refresh_predictions(tickers, include_text=include_text)
                logger.info("Scheduler refreshed predictions: %s", outcomes)
        except Exception:
            logger.exception("Scheduler cycle failed unexpectedly")
        await asyncio.sleep(interval_seconds)


def scheduler_config_from_env() -> dict:
    """Real quota math behind the default interval: this scheduler's default prediction
    refresh spends no News/YouTube quota (include_text defaults to False), only CPU time
    (~10-30s/ticker * 5 tickers ~= 1-3 minutes/cycle) -- safe at any interval. If
    SCHEDULER_INCLUDE_TEXT=true, each cycle spends real quota: YouTube's
    get_recent_uploads costs ~2 units/channel (2 channels = 4 units/cycle) and NewsAPI
    costs 1 request per (ticker, query) combination against its 100 requests/day cap --
    at 5 tickers that's 5 requests/cycle, so more than ~20 cycles/day would exceed
    NewsAPI's free-tier daily cap. The default 6-hour interval (4 cycles/day) stays
    comfortably under that even with text enabled.
    """
    # Defense-in-depth: never auto-start real training/network calls under pytest, even
    # if something in the future constructs the FastAPI app in a way that DOES trigger
    # the lifespan during tests (today it does not -- confirmed directly, TestClient(app)
    # is used without `with` in tests/unit/test_api.py, so lifespan never runs there --
    # but a future test change could alter that, and a scheduler silently kicking off
    # real 10-30s training runs + real API quota usage during a test suite would be a
    # serious, hard-to-diagnose regression).
    if "PYTEST_CURRENT_TEST" in os.environ:
        return {
            "enabled": False, "interval_seconds": 21600, "tickers": [],
            "refresh_health": False, "refresh_predictions": False,
            "health_live_probe": False, "include_text": False,
        }
    return {
        "enabled": os.getenv("SCHEDULER_ENABLED", "true").lower() == "true",
        "interval_seconds": int(os.getenv("SCHEDULER_INTERVAL_SECONDS", "21600")),  # 6 hours
        "tickers": [t.strip().upper() for t in os.getenv("SCHEDULER_TICKERS", ",".join(DEFAULT_TICKERS)).split(",") if t.strip()],
        "refresh_health": os.getenv("SCHEDULER_REFRESH_DATA_HEALTH", "true").lower() == "true",
        "refresh_predictions": os.getenv("SCHEDULER_REFRESH_PREDICTIONS", "true").lower() == "true",
        "health_live_probe": os.getenv("SCHEDULER_HEALTH_LIVE_PROBE", "false").lower() == "true",
        "include_text": os.getenv("SCHEDULER_INCLUDE_TEXT", "false").lower() == "true",
    }
