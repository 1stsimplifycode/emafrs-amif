"""yfinance client: OHLCV, returns, and adjusted-close history. No API key required.

Provenance note: yfinance scrapes Yahoo Finance's undocumented endpoints. It is free and
requires no key, but has no formal SLA/rate-limit guarantee — treated here as the primary
market-data source with Alpha Vantage as a validation/fallback source (see
alphavantage_client.py and docs/DATA_SOURCES_AND_APIS.md).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

from emafrs.data.schema import SourceRecord

logger = logging.getLogger("emafrs.data.yfinance")

LICENSE_NOTE = (
    "Yahoo Finance data via yfinance (unofficial). Free for research/personal use; "
    "no redistribution guarantee; no formal SLA. Not for high-frequency/commercial use."
)


class YFinanceClient:
    provider = "yfinance"

    def get_ohlcv(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> tuple[pd.DataFrame, SourceRecord]:
        """Returns (dataframe indexed by date with OHLCV + adj close, provenance record).

        Raises ValueError if yfinance returns no rows (delisted ticker, bad symbol, or
        outage) -- callers must surface this rather than substituting synthetic data.
        """
        fetched_at = datetime.now(timezone.utc)
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if df.empty:
            raise ValueError(
                f"yfinance returned no data for ticker={ticker!r} period={period!r} interval={interval!r}"
            )
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        df.index.name = "date"
        df = df.tz_localize(None) if df.index.tz is not None else df

        record = SourceRecord(
            provider=self.provider,
            endpoint="ohlcv_history",
            fetched_at=fetched_at,
            observation_time=df.index.max().to_pydatetime().replace(tzinfo=timezone.utc),
            entity=ticker,
            fields={"n_rows": len(df), "period": period, "interval": interval,
                    "first_date": str(df.index.min()), "last_date": str(df.index.max())},
            license_note=LICENSE_NOTE,
            cache_hit=False,
        )
        return df, record

    def get_intraday_1m(self, ticker: str) -> tuple[pd.DataFrame, SourceRecord]:
        """1-minute bars. Yahoo restricts 1m history to the trailing 7 days regardless of
        account tier -- used only as a partial substitute for true tick data in the MMIE
        proxy signals (see docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md row #4)."""
        return self.get_ohlcv(ticker, period="7d", interval="1m")
