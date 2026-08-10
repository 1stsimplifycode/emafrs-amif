"""Tests for the MMIE's OHLCV-bar proxy signals: correctness against known cases, and
no-lookahead leakage audits -- including a regression test for the whole-series-bucketing
VPIN bug found and fixed during implementation (see proxies.py's vpin_proxy docstring)."""
import numpy as np
import pandas as pd
import pytest

from emafrs.features.leakage_audit import assert_no_lookahead
from emafrs.microstructure.proxies import (
    alpha_decay,
    alpha_persistence,
    build_microstructure_features,
    delta_obi_proxy,
    depth_depletion_rate,
    liquidity_shock_proxy,
    obi_proxy,
    ofi_proxy,
    vpin_proxy,
)


def _synthetic_ohlcv(n: int = 250, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="min")
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    high = close + rng.uniform(0.01, 0.3, n)
    low = close - rng.uniform(0.01, 0.3, n)
    open_ = close + rng.normal(0, 0.05, n)
    volume = rng.integers(1000, 10000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates)


def test_obi_proxy_extremes():
    high = pd.Series([10.0, 10.0])
    low = pd.Series([8.0, 8.0])
    close_at_high = pd.Series([10.0, 8.0])  # bar 0 closes at the high, bar 1 at the low
    obi = obi_proxy(high, low, close_at_high)
    assert obi.iloc[0] == pytest.approx(1.0)   # closed at high -> max buy pressure
    assert obi.iloc[1] == pytest.approx(-1.0)  # closed at low -> max sell pressure


def test_obi_proxy_bounded_minus_one_to_one():
    df = _synthetic_ohlcv()
    obi = obi_proxy(df["high"], df["low"], df["close"])
    valid = obi.dropna()
    assert (valid >= -1 - 1e-9).all() and (valid <= 1 + 1e-9).all()


def test_delta_obi_proxy_is_first_difference():
    obi = pd.Series([0.1, 0.5, -0.2, 0.0])
    delta = delta_obi_proxy(obi)
    assert np.isnan(delta.iloc[0])
    assert delta.iloc[1] == pytest.approx(0.4)
    assert delta.iloc[2] == pytest.approx(-0.7)


def test_ofi_proxy_signs_match_price_direction():
    close = pd.Series([100.0, 101.0, 100.5, 100.5, 102.0])
    volume = pd.Series([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
    ofi = ofi_proxy(close, volume)
    assert ofi.iloc[1] > 0   # price rose -> buy-signed
    assert ofi.iloc[2] < 0   # price fell -> sell-signed
    assert ofi.iloc[3] == ofi.iloc[2]  # flat bar inherits prior direction (still negative)
    assert ofi.iloc[4] > 0


def test_liquidity_shock_proxy_zero_at_baseline_and_positive_on_spike():
    n = 60
    high = pd.Series([101.0] * n)
    low = pd.Series([99.0] * n)
    close = pd.Series([100.0] * n)
    # inject a range spike at the end
    high.iloc[-1] = 110.0
    lsi = liquidity_shock_proxy(high, low, close, window=21)
    assert lsi.iloc[-2] < 0.1     # stable range -> near-zero deviation from baseline
    assert lsi.iloc[-1] > 1.0     # spike -> large deviation


def test_depth_depletion_rate_is_deliberately_unimplemented():
    with pytest.raises(NotImplementedError):
        depth_depletion_rate()


def test_alpha_persistence_high_when_signal_predicts_forward_return():
    n = 200
    rng = np.random.default_rng(3)
    signal = pd.Series(rng.normal(0, 1, n))
    forward_returns = signal * 0.8 + pd.Series(rng.normal(0, 0.1, n))  # strongly correlated
    ap = alpha_persistence(signal, forward_returns, window=63)
    assert ap.dropna().mean() > 0.5


def test_alpha_decay_shrinks_toward_zero_over_time():
    alpha = pd.Series([1.0] * 5)
    decayed = alpha_decay(alpha, decay_rate=0.5, elapsed_periods=pd.Series([0, 1, 2, 3, 4]))
    assert decayed.is_monotonic_decreasing
    assert decayed.iloc[0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "feature_fn",
    [
        lambda d: obi_proxy(d["high"], d["low"], d["close"]),
        lambda d: delta_obi_proxy(obi_proxy(d["high"], d["low"], d["close"])),
        lambda d: liquidity_shock_proxy(d["high"], d["low"], d["close"], window=21),
        lambda d: ofi_proxy(d["close"], d["volume"]),
    ],
)
def test_bar_level_proxies_pass_no_lookahead_check(feature_fn):
    df = _synthetic_ohlcv()
    result = assert_no_lookahead(feature_fn, df, n_samples=10, min_index=40)
    assert result.passed, result.details


def test_vpin_proxy_passes_no_lookahead_check_with_trailing_window():
    """Regression test for the leakage bug found during implementation: an earlier
    version bucketed by the WHOLE series' total volume (cum_vol.iloc[-1] / n_buckets),
    which made every output depend on future bars. This test would have failed against
    that version and must keep passing against the trailing-window fix."""
    df = _synthetic_ohlcv(n=200)
    result = assert_no_lookahead(
        lambda d: vpin_proxy(d["high"], d["low"], d["close"], d["volume"], window=50, n_buckets=10),
        df, n_samples=10, min_index=60,
    )
    assert result.passed, result.details


def test_build_microstructure_features_returns_expected_columns():
    df = _synthetic_ohlcv()
    feats = build_microstructure_features(df, vol_window=21)
    expected = {"obi_proxy", "delta_obi_proxy", "liquidity_shock_proxy", "ofi_proxy", "vpin_proxy", "volume_anomaly"}
    assert expected.issubset(set(feats.columns))
    assert len(feats) == len(df)
