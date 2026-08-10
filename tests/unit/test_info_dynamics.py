import numpy as np
import pandas as pd

from emafrs.regime.info_dynamics import dfa_alpha, hurst_rs, rolling_entropy, rolling_hurst, shannon_entropy


def test_hurst_of_pure_random_walk_increments_near_half():
    """White noise (i.i.d. increments) should give H close to 0.5 (random walk)."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 2000)
    h = hurst_rs(x, min_chunk=8, n_chunk_sizes=15)
    assert 0.35 < h < 0.65


def test_hurst_increases_with_ar1_persistence():
    """A positively-autocorrelated (momentum/persistent) AR(1) series should get a higher
    R/S Hurst estimate than i.i.d. white noise with the same innovation variance."""
    rng = np.random.default_rng(1)
    n = 2000
    eps = rng.normal(0, 1, n)

    def ar1(phi: float) -> np.ndarray:
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + eps[t]
        return x

    h_white = hurst_rs(ar1(0.0), min_chunk=8, n_chunk_sizes=15)
    h_persistent = hurst_rs(ar1(0.9), min_chunk=8, n_chunk_sizes=15)
    assert h_persistent > h_white


def test_hurst_returns_nan_for_too_short_series():
    assert np.isnan(hurst_rs(np.array([1.0, 2.0, 3.0]), min_chunk=8))


def test_dfa_alpha_of_random_walk_increments_near_half():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 2000)
    alpha = dfa_alpha(x, min_box=4, n_box_sizes=15)
    assert 0.35 < alpha < 0.65


def test_shannon_entropy_uniform_higher_than_concentrated_over_same_range():
    """shannon_entropy() auto-ranges its histogram to each sample's own min/max, so it is
    shape-sensitive, not scale-sensitive (a rescaled Gaussian has ~the same entropy as the
    original). The correct sensitivity check is shape at a shared range: a uniform
    distribution maximizes entropy for bounded support, so it must score higher than a
    concentrated (peaked) distribution clipped to the same range."""
    rng = np.random.default_rng(3)
    n_bins = 20
    uniform = rng.uniform(-1, 1, 5000)
    concentrated = np.clip(rng.normal(0, 0.15, 5000), -1, 1)

    h_uniform = shannon_entropy(uniform, n_bins=n_bins)
    h_concentrated = shannon_entropy(concentrated, n_bins=n_bins)
    assert h_uniform > h_concentrated
    assert h_uniform > np.log2(n_bins) - 0.5  # near the theoretical max for a uniform histogram


def test_shannon_entropy_of_constant_series_is_low():
    x = np.zeros(100)
    h = shannon_entropy(x, n_bins=10)
    # all mass in one bin -> entropy 0
    assert h == 0.0


def test_rolling_hurst_and_entropy_produce_expected_nan_prefix():
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    rng = np.random.default_rng(4)
    returns = pd.Series(rng.normal(0, 0.01, 300), index=dates)

    h = rolling_hurst(returns, window=126, min_chunk=8)
    assert h.iloc[:125].isna().all()
    assert h.iloc[125:].notna().sum() > 0

    e = rolling_entropy(returns, window=21)
    assert e.iloc[:20].isna().all()
    assert e.iloc[20:].notna().all()
