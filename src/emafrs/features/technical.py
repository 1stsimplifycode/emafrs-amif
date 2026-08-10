"""Technical indicators computed from OHLCV data, matching PDF spec Section 6.2.2-6.2.3.

All computations use pandas `.rolling()`/`.ewm()`, which by construction only reference the
current row and earlier rows (no `center=True`, no negative shifts) -- this is what makes
them leakage-safe. See `emafrs.features.leakage_audit.assert_no_lookahead` for the automated
check that actually verifies this property rather than merely asserting it in a comment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(close: pd.Series) -> pd.Series:
    """Daily arithmetic return r_t = (P_t - P_{t-1}) / P_{t-1}. PDF Eq. (unnumbered, Sec 6.2.2)."""
    return close.pct_change()


def compute_cumulative_return(close: pd.Series, window: int = 21) -> pd.Series:
    return close.pct_change(periods=window)


def compute_realized_volatility(returns: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series:
    vol = returns.rolling(window, min_periods=window).std()
    return vol * np.sqrt(252) if annualize else vol


def compute_max_drawdown(close: pd.Series, window: int = 252) -> pd.Series:
    """Drawdown from the rolling `window`-period peak, evaluated at each t using only
    close[t-window+1 : t+1] -- never a peak formed later than t."""
    rolling_peak = close.rolling(window, min_periods=1).max()
    return (close - rolling_peak) / rolling_peak


def compute_sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def compute_ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI. Uses only past `window` changes at each t (via ewm with alpha=1/window,
    which is mathematically a causal exponential filter)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100.0).where(avg_loss != 0, 100.0).where(avg_gain != 0, rsi)


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd - signal_line
    return pd.DataFrame({"macd": macd, "macd_signal": signal_line, "macd_hist": hist})


def compute_bollinger_bands(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    pct_b = (close - lower) / (upper - lower)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_pct_b": pct_b})


def compute_all_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """df must have a 'close' column (and ideally open/high/low/volume), indexed by date
    ascending. Returns a new DataFrame of features aligned to the same index; early rows
    are NaN where a window hasn't filled yet (never forward/back-filled here -- that is a
    downstream, explicitly-documented preprocessing decision, not silently done in feature
    computation)."""
    close = df["close"]
    returns = compute_returns(close)

    out = pd.DataFrame(index=df.index)
    out["return_1d"] = returns
    out["cumulative_return_21d"] = compute_cumulative_return(close, 21)
    out["realized_vol_21d"] = compute_realized_volatility(returns, 21)
    out["max_drawdown_252d"] = compute_max_drawdown(close, 252)
    out["sma_10"] = compute_sma(close, 10)
    out["sma_50"] = compute_sma(close, 50)
    out["ema_12"] = compute_ema(close, 12)
    out["ema_26"] = compute_ema(close, 26)
    out["rsi_14"] = compute_rsi(close, 14)

    macd_df = compute_macd(close)
    out = out.join(macd_df)

    bb_df = compute_bollinger_bands(close)
    out = out.join(bb_df)

    if "volume" in df.columns:
        vol_mean = df["volume"].rolling(21, min_periods=21).mean()
        vol_std = df["volume"].rolling(21, min_periods=21).std()
        out["volume_zscore_21d"] = (df["volume"] - vol_mean) / vol_std

    return out
