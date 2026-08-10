"""Portfolio/risk indicators with STANDARD, well-defined formulas -- PDF Section 6.2.5
names Conditional Value at Risk (CVaR), Value-at-Risk (VaR), and Residual Volatility, all
of which have unambiguous standard finance definitions independent of the patent text, so
they are implemented directly (unlike AMSF/FTCS/Adaptive Signal Weighting -- see the note
in docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md: those are named in the PDF as ingested
outputs of unspecified "ancillary algorithmic scoring systems" with no formula given
anywhere in the specification, so they are NOT implemented here rather than inventing a
formula and presenting it as if it were the system's own defined signal).

All rolling -- causal by construction, same convention as features/technical.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_historical_var(returns: pd.Series, window: int = 63, confidence: float = 0.95) -> pd.Series:
    """Historical-simulation Value-at-Risk: the `1-confidence` empirical quantile of
    returns over the trailing window, expressed as a positive loss magnitude."""
    q = 1 - confidence
    return -returns.rolling(window, min_periods=window).quantile(q)


def rolling_cvar(returns: pd.Series, window: int = 63, confidence: float = 0.95) -> pd.Series:
    """Conditional Value-at-Risk (Expected Shortfall): mean of returns at or below the
    VaR quantile within the trailing window, expressed as a positive loss magnitude."""
    q = 1 - confidence

    def _cvar(x: np.ndarray) -> float:
        threshold = np.quantile(x, q)
        tail = x[x <= threshold]
        return -tail.mean() if len(tail) else np.nan

    return returns.rolling(window, min_periods=window).apply(_cvar, raw=True)


def rolling_residual_volatility(
    asset_returns: pd.Series, benchmark_returns: pd.Series, window: int = 63
) -> pd.Series:
    """Idiosyncratic volatility unexplained by the benchmark: residual std from a rolling
    OLS of asset returns on benchmark returns (single-factor market model), computed
    causally per window (no look-ahead: each window's beta is estimated only from data
    within that trailing window)."""
    aligned = pd.concat([asset_returns.rename("asset"), benchmark_returns.rename("bench")], axis=1).dropna()

    def _resid_vol(sub: pd.DataFrame) -> float:
        x = sub["bench"].to_numpy()
        y = sub["asset"].to_numpy()
        if x.std() == 0:
            return np.nan
        beta, alpha = np.polyfit(x, y, 1)
        resid = y - (alpha + beta * x)
        return float(np.std(resid))

    out = pd.Series(np.nan, index=aligned.index)
    values = aligned.to_numpy()
    for i in range(window - 1, len(aligned)):
        sub = aligned.iloc[i - window + 1 : i + 1]
        out.iloc[i] = _resid_vol(sub)
    return out.reindex(asset_returns.index)


def rolling_var_change(var_series: pd.Series, lag: int = 1) -> pd.Series:
    """Rolling change in VaR -- PDF Sec 6.2.5's "rolling Value-at-Risk (VaR) change metrics"."""
    return var_series.diff(lag)
