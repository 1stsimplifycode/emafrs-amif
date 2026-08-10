import numpy as np
import pandas as pd
import pytest

from emafrs.features.leakage_audit import assert_no_lookahead
from emafrs.features.risk_metrics import (
    rolling_cvar,
    rolling_historical_var,
    rolling_residual_volatility,
    rolling_var_change,
)


def _returns_series(n=300, seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0, 0.01, n), index=dates)


def test_var_and_cvar_are_positive_for_typical_returns():
    r = _returns_series()
    var95 = rolling_historical_var(r, window=63)
    cvar95 = rolling_cvar(r, window=63)
    assert (var95.dropna() > 0).all()
    assert (cvar95.dropna() > 0).all()


def test_cvar_at_least_as_large_as_var():
    """CVaR (expected shortfall beyond the VaR threshold) must be >= VaR by definition."""
    r = _returns_series()
    var95 = rolling_historical_var(r, window=63)
    cvar95 = rolling_cvar(r, window=63)
    diff = (cvar95 - var95).dropna()
    assert (diff >= -1e-9).all()


def test_residual_volatility_lower_when_returns_track_benchmark_closely():
    n = 200
    rng = np.random.default_rng(6)
    bench = pd.Series(rng.normal(0, 0.01, n), index=pd.date_range("2023-01-01", periods=n, freq="B"))
    tracking = bench * 1.0 + pd.Series(rng.normal(0, 0.0005, n), index=bench.index)  # tight tracker
    noisy = bench * 1.0 + pd.Series(rng.normal(0, 0.02, n), index=bench.index)  # loose tracker

    resid_tight = rolling_residual_volatility(tracking, bench, window=63).dropna().mean()
    resid_noisy = rolling_residual_volatility(noisy, bench, window=63).dropna().mean()
    assert resid_tight < resid_noisy


def test_var_change_is_first_difference():
    var = pd.Series([0.01, 0.015, 0.012])
    change = rolling_var_change(var)
    assert np.isnan(change.iloc[0])
    assert change.iloc[1] == pytest.approx(0.005)


@pytest.mark.parametrize(
    "feature_fn",
    [
        lambda d: rolling_historical_var(d["r"], window=63),
        lambda d: rolling_cvar(d["r"], window=63),
    ],
)
def test_var_cvar_pass_no_lookahead_check(feature_fn):
    r = _returns_series()
    df = r.to_frame("r")
    result = assert_no_lookahead(feature_fn, df, n_samples=8, min_index=70)
    assert result.passed, result.details
