"""Integration test: hits live Yahoo Finance via yfinance. Requires network access.
No API key needed. Skips gracefully if the network/provider is unreachable rather than
failing the whole suite, since this is a genuine external dependency, not project code."""
import pytest

from emafrs.data.yfinance_client import YFinanceClient


@pytest.mark.integration
def test_get_ohlcv_returns_real_recent_data():
    try:
        df, record = YFinanceClient().get_ohlcv("AAPL", period="1mo", interval="1d")
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"yfinance unreachable in this environment: {exc}")

    assert not df.empty
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)
    assert record.provider == "yfinance"
    assert record.entity == "AAPL"
    assert record.cache_hit is False
    assert (df["high"] >= df["low"]).all()


@pytest.mark.integration
def test_unknown_ticker_raises_instead_of_returning_empty():
    with pytest.raises(ValueError):
        YFinanceClient().get_ohlcv("THIS_IS_NOT_A_REAL_TICKER_XYZ123", period="5d")
