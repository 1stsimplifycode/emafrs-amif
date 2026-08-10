"""Tests for the leakage audit itself -- including a deliberately leaky feature function
to prove the no-lookahead check actually catches lookahead, not just that it passes on
well-behaved input (a check that only ever says "pass" is worthless)."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from emafrs.features import technical
from emafrs.features.leakage_audit import (
    assert_no_lookahead,
    check_chronological_split,
    check_news_before_prediction_time,
    check_rolling_zscore_uses_only_past,
)


def _synthetic_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates
    )


def test_centered_rolling_mean_fails_no_lookahead_check():
    """A deliberately leaky feature (centered rolling window) must be CAUGHT, proving the
    audit has teeth rather than trivially passing everything."""
    df = _synthetic_ohlcv()

    def leaky_feature(d: pd.DataFrame) -> pd.Series:
        return d["close"].rolling(21, center=True, min_periods=1).mean()

    result = assert_no_lookahead(leaky_feature, df, n_samples=10, min_index=60)
    assert result.passed is False
    assert len(result.details["mismatches"]) > 0


def test_shift_negative_fails_no_lookahead_check():
    """Using tomorrow's value today (a classic bug) must also be caught."""
    df = _synthetic_ohlcv()

    def leaky_feature(d: pd.DataFrame) -> pd.Series:
        return d["close"].shift(-1)  # uses the NEXT row's value

    result = assert_no_lookahead(leaky_feature, df, n_samples=10, min_index=60)
    assert result.passed is False


@pytest.mark.parametrize(
    "feature_fn",
    [
        lambda d: technical.compute_returns(d["close"]),
        lambda d: technical.compute_sma(d["close"], 10),
        lambda d: technical.compute_ema(d["close"], 12),
        lambda d: technical.compute_rsi(d["close"], 14),
        lambda d: technical.compute_macd(d["close"])["macd_hist"],
        lambda d: technical.compute_bollinger_bands(d["close"])["bb_pct_b"],
        lambda d: technical.compute_realized_volatility(technical.compute_returns(d["close"]), 21),
        lambda d: technical.compute_max_drawdown(d["close"], 252),
    ],
)
def test_real_technical_features_pass_no_lookahead_check(feature_fn):
    df = _synthetic_ohlcv()
    result = assert_no_lookahead(feature_fn, df, n_samples=10, min_index=60)
    assert result.passed, result.details


def test_rolling_zscore_passes_no_lookahead_check():
    df = _synthetic_ohlcv()
    result = check_rolling_zscore_uses_only_past(df["close"], window=63)
    assert result.passed, result.details


def test_news_before_prediction_time_flags_future_article():
    prediction_time = datetime(2024, 1, 10, tzinfo=timezone.utc)
    records = [
        {"publishedAt": "2024-01-09T12:00:00Z"},  # ok
        {"publishedAt": "2024-01-11T00:00:00Z"},  # violation: after prediction time
    ]
    result = check_news_before_prediction_time(records, prediction_time)
    assert result.passed is False
    assert result.details["violations"]


def test_news_before_prediction_time_passes_when_all_past():
    prediction_time = datetime(2024, 1, 10, tzinfo=timezone.utc)
    records = [
        {"publishedAt": "2024-01-09T12:00:00Z"},
        {"publishedAt": "2024-01-08T00:00:00Z"},
    ]
    result = check_news_before_prediction_time(records, prediction_time)
    assert result.passed is True


def test_chronological_split_detects_overlap_and_order_violation():
    train = pd.date_range("2023-01-01", "2023-06-30", freq="B")
    test_bad = pd.date_range("2023-06-01", "2023-07-31", freq="B")  # overlaps train
    result = check_chronological_split(train, test_bad)
    assert result.passed is False

    test_good = pd.date_range("2023-07-01", "2023-08-31", freq="B")
    result_good = check_chronological_split(train, test_good)
    assert result_good.passed is True
