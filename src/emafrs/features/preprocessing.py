"""Leakage-safe preprocessing: rolling z-score standardization and calendar alignment.

PDF Section 6.3: "All numerical financial and indicator variables undergo z-score
standardization within rolling temporal windows." The key word is ROLLING -- standardizing
against the full-sample mean/std (as many tutorials do) leaks the future distribution into
early-period features. Every transform here is a strict function of data at or before t.
"""
from __future__ import annotations

import pandas as pd


def rolling_zscore(series: pd.Series, window: int = 63, min_periods: int | None = None) -> pd.Series:
    """z_t = (x_t - mean(x_{t-window+1..t})) / std(x_{t-window+1..t}).

    Both the mean and std at t are computed from a window ending at t (pandas `.rolling()`
    default, no `center=True`), so z_t never depends on x_{t+1..}. min_periods defaults to
    the full window so early points are NaN rather than standardized against a partial,
    noisy window.
    """
    min_periods = min_periods or window
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std


def rolling_zscore_frame(df: pd.DataFrame, window: int = 63, columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or list(df.columns)
    out = df.copy()
    for col in columns:
        out[col] = rolling_zscore(df[col], window=window)
    return out


def align_to_business_calendar(df: pd.DataFrame, freq: str = "B") -> pd.DataFrame:
    """Reindex to a business-day calendar and forward-fill gaps from market closures
    (weekends/holidays). Forward-fill only ever propagates a PAST value forward -- it
    cannot introduce future information, unlike back-fill (which this function never uses).
    """
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    return df.reindex(full_index).ffill()
