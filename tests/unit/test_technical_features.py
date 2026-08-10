"""Correctness tests for technical indicators (separate from leakage tests)."""
import numpy as np
import pandas as pd
import pytest

from emafrs.features import technical


def _flat_then_rising_close(n: int = 80) -> pd.Series:
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    values = np.concatenate([np.full(20, 100.0), np.linspace(100, 150, n - 20)])
    return pd.Series(values, index=dates)


def test_sma_matches_manual_mean():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    sma3 = technical.compute_sma(close, 3)
    assert np.isnan(sma3.iloc[0])
    assert np.isnan(sma3.iloc[1])
    assert sma3.iloc[2] == pytest.approx(2.0)
    assert sma3.iloc[4] == pytest.approx(4.0)


def test_rsi_is_bounded_0_100():
    close = _flat_then_rising_close()
    rsi = technical.compute_rsi(close, 14)
    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_near_100_for_strictly_rising_series():
    close = pd.Series(np.linspace(100, 200, 60))
    rsi = technical.compute_rsi(close, 14)
    assert rsi.dropna().iloc[-1] > 95


def test_macd_histogram_equals_macd_minus_signal():
    close = _flat_then_rising_close()
    macd_df = technical.compute_macd(close)
    diff = (macd_df["macd"] - macd_df["macd_signal"]) - macd_df["macd_hist"]
    assert diff.dropna().abs().max() < 1e-9


def test_bollinger_bands_upper_above_lower():
    close = _flat_then_rising_close()
    bb = technical.compute_bollinger_bands(close)
    valid = bb.dropna()
    assert (valid["bb_upper"] >= valid["bb_lower"]).all()


def test_max_drawdown_is_non_positive():
    close = _flat_then_rising_close()
    dd = technical.compute_max_drawdown(close, window=252)
    assert (dd.dropna() <= 1e-9).all()


def test_compute_all_technical_features_returns_expected_columns():
    n = 120
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "open": np.random.default_rng(0).normal(100, 1, n),
            "high": np.random.default_rng(1).normal(101, 1, n),
            "low": np.random.default_rng(2).normal(99, 1, n),
            "close": np.random.default_rng(3).normal(100, 1, n),
            "volume": np.random.default_rng(4).integers(1e6, 5e6, n),
        },
        index=dates,
    )
    feats = technical.compute_all_technical_features(df)
    expected = {
        "return_1d", "cumulative_return_21d", "realized_vol_21d", "max_drawdown_252d",
        "sma_10", "sma_50", "ema_12", "ema_26", "rsi_14",
        "macd", "macd_signal", "macd_hist", "bb_mid", "bb_upper", "bb_lower", "bb_pct_b",
        "volume_zscore_21d",
    }
    assert expected.issubset(set(feats.columns))
    assert len(feats) == n
