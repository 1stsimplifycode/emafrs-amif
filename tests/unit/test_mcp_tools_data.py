"""Unit tests for the helper MCP server's data-access tool wrappers.

Unconfigured-source tests need no network. The yfinance test hits live network (no key
required) and is marked integration accordingly.
"""
import pytest

from emafrs.mcp_server import tools_data


def test_unconfigured_source_returns_ok_false_not_exception():
    result = tools_data.get_quote("AAPL")  # FINNHUB_API_KEY assumed unset in CI
    assert result["ok"] is False
    assert "FINNHUB_API_KEY" in result["error"]


def test_unconfigured_newsapi_returns_ok_false():
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
