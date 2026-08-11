"""Unit tests for emafrs.mlops.scheduler -- the real background job that was referenced
in docstrings elsewhere in this codebase but never actually implemented until now. These
tests mock the real training/health-check calls (which are separately, genuinely
exercised in tests/integration/test_mlops_pipeline.py and via the live API checks) to
test the scheduler's own orchestration logic in isolation: does it call the right
functions, does one ticker's failure not kill the others, does the env-var config parse
correctly."""
import os
from unittest.mock import MagicMock, patch

import pytest

from emafrs.mlops import scheduler


def test_scheduler_config_from_env_uses_real_defaults(monkeypatch):
    for key in [
        "SCHEDULER_ENABLED", "SCHEDULER_INTERVAL_SECONDS", "SCHEDULER_TICKERS",
        "SCHEDULER_REFRESH_DATA_HEALTH", "SCHEDULER_REFRESH_PREDICTIONS",
        "SCHEDULER_HEALTH_LIVE_PROBE", "SCHEDULER_INCLUDE_TEXT",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    cfg = scheduler.scheduler_config_from_env()
    assert cfg["enabled"] is True
    assert cfg["interval_seconds"] == 21600
    assert cfg["tickers"] == scheduler.DEFAULT_TICKERS
    assert cfg["health_live_probe"] is False  # quota-safe default
    assert cfg["include_text"] is False  # quota-safe default


def test_scheduler_config_from_env_respects_overrides(monkeypatch):
    monkeypatch.setenv("SCHEDULER_ENABLED", "false")
    monkeypatch.setenv("SCHEDULER_INTERVAL_SECONDS", "60")
    monkeypatch.setenv("SCHEDULER_TICKERS", "spy, qqq ,  aapl")
    monkeypatch.setenv("SCHEDULER_HEALTH_LIVE_PROBE", "true")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    cfg = scheduler.scheduler_config_from_env()
    assert cfg["enabled"] is False
    assert cfg["interval_seconds"] == 60
    assert cfg["tickers"] == ["SPY", "QQQ", "AAPL"]  # trimmed and uppercased
    assert cfg["health_live_probe"] is True


def test_scheduler_config_forces_disabled_under_pytest(monkeypatch):
    """Defense-in-depth: even if every other env var says "enabled", running under
    pytest must never auto-start real training/network calls."""
    monkeypatch.setenv("SCHEDULER_ENABLED", "true")
    assert os.environ.get("PYTEST_CURRENT_TEST") is not None  # pytest sets this itself
    cfg = scheduler.scheduler_config_from_env()
    assert cfg["enabled"] is False


def test_refresh_predictions_reports_per_ticker_outcome_and_does_not_stop_on_one_failure():
    with patch("emafrs.mlops.training_pipeline.run_training_and_prediction_pipeline") as mock_run:
        def side_effect(ticker, include_text=False):
            if ticker == "BAD":
                raise RuntimeError("simulated provider outage")
            record = MagicMock()
            record.direction = 1
            record.confidence = 0.55
            return record

        mock_run.side_effect = side_effect
        outcomes = scheduler.refresh_predictions(["SPY", "BAD", "QQQ"])

    assert "ok" in outcomes["SPY"]
    assert "error" in outcomes["BAD"]
    assert "ok" in outcomes["QQQ"]
    assert mock_run.call_count == 3  # BAD's failure did not stop SPY/QQQ from being attempted


def test_refresh_predictions_passes_include_text_through():
    with patch("emafrs.mlops.training_pipeline.run_training_and_prediction_pipeline") as mock_run:
        record = MagicMock(direction=1, confidence=0.5)
        mock_run.return_value = record
        scheduler.refresh_predictions(["SPY"], include_text=True)
    mock_run.assert_called_once_with("SPY", include_text=True)


def test_refresh_data_health_persists_one_record_per_provider():
    fake_health = {
        "yfinance": MagicMock(configured=True, reachable=True, last_success_at=None, last_error=None),
        "newsapi": MagicMock(configured=False, reachable=None, last_success_at=None, last_error=None),
    }
    with patch("emafrs.data.health.check_all", return_value=fake_health) as mock_check, \
         patch("emafrs.db.session.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        n = scheduler.refresh_data_health(live=False)

    mock_check.assert_called_once_with(live=False)
    assert n == 2
    assert mock_session.add.call_count == 2
    mock_session.commit.assert_called_once()
    mock_session.close.assert_called_once()


@pytest.mark.asyncio
async def test_run_scheduler_loop_runs_at_least_one_cycle_before_cancellation():
    import asyncio

    calls = []

    async def fake_sleep(_seconds):
        calls.append("slept")
        raise asyncio.CancelledError()  # stop after the first cycle, like a real shutdown would

    with patch("emafrs.mlops.scheduler.refresh_data_health", return_value=1) as mock_health, \
         patch("emafrs.mlops.scheduler.refresh_predictions", return_value={"SPY": "ok"}) as mock_preds, \
         patch("asyncio.sleep", side_effect=fake_sleep):
        with pytest.raises(asyncio.CancelledError):
            await scheduler.run_scheduler_loop(interval_seconds=1, tickers=["SPY"])

    mock_health.assert_called_once()
    mock_preds.assert_called_once()
    assert calls == ["slept"]
