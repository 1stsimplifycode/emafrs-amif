"""Unit tests for the helper MCP server's data-access tool wrappers.

Unconfigured-source tests explicitly monkeypatch CONFIG rather than relying on ambient
.env state -- a real test failure during development showed why: once a real (if invalid)
NEWS_API_KEY was added to .env, a test that implicitly assumed "no key is set in this
environment" broke, because it was testing environment state rather than the tool's own
"ok: False when unconfigured" behavior in isolation. The yfinance test hits live network
(no key required) and is marked integration accordingly.
"""
import pytest

from emafrs.data.config import DataSourceConfig
from emafrs.mcp_server import tools_data


def _unconfigured_config() -> DataSourceConfig:
    return DataSourceConfig(
        youtube_api_key=None, news_api_key=None, alphavantage_api_key=None,
        finnhub_api_key=None, fred_api_key=None,
    )


def test_unconfigured_source_returns_ok_false_not_exception(monkeypatch):
    monkeypatch.setattr(tools_data, "CONFIG", _unconfigured_config())
    result = tools_data.get_quote("AAPL")
    assert result["ok"] is False
    assert "FINNHUB_API_KEY" in result["error"]


def test_unconfigured_newsapi_returns_ok_false(monkeypatch):
    monkeypatch.setattr(tools_data, "CONFIG", _unconfigured_config())
    result = tools_data.search_news("Apple")
    assert result["ok"] is False
    assert "NEWS_API_KEY" in result["error"]


@pytest.mark.integration
def test_get_ohlcv_tool_returns_rows_with_provenance_meta():
    result = tools_data.get_ohlcv("AAPL", period="1mo")
    if not result["ok"]:
        pytest.skip(f"yfinance unreachable in this environment: {result['error']}")
    assert result["rows"]
    assert result["meta"]["provider"] == "yfinance"
    assert result["meta"]["cache_hit"] is False


@pytest.mark.integration
def test_sec_ticker_to_cik_tool_handles_failure_gracefully():
    # May legitimately fail (SEC bot-defense / rate limiting observed during development);
    # the contract under test is that it returns a structured result, never raises.
    result = tools_data.sec_ticker_to_cik("AAPL")
    assert "ok" in result
