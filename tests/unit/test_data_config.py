"""Unit tests for DAL configuration detection: no network access."""
from emafrs.data.config import DataSourceConfig


def test_yfinance_and_sec_edgar_need_no_key():
    cfg = DataSourceConfig(
        youtube_api_key=None, news_api_key=None, alphavantage_api_key=None,
        finnhub_api_key=None, fred_api_key=None,
    )
    sources = cfg.configured_sources()
    assert sources["yfinance"] is True
    assert sources["sec_edgar"] is True
    assert sources["alphavantage"] is False
    assert sources["finnhub"] is False
    assert sources["newsapi"] is False
    assert sources["youtube"] is False
    assert sources["fred"] is False


def test_keyed_sources_become_configured_when_key_present():
    cfg = DataSourceConfig(
        youtube_api_key="yk", news_api_key="nk", alphavantage_api_key="ak",
        finnhub_api_key="fk", fred_api_key="frk",
    )
    sources = cfg.configured_sources()
    assert all(sources.values())
