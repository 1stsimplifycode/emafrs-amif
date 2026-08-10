"""Tests that build_daily_dataset_from_ohlcv correctly joins a pre-computed text_daily
DataFrame (synthetic here -- the real fetch is exercised in
tests/integration/test_youtube_text_dataset.py) using the same masking pattern already
proven for microstructure. No network required: OHLCV is synthetic."""
import numpy as np
import pandas as pd
import pytest

from emafrs.models.dataset import build_daily_dataset_from_ohlcv
from emafrs.nlp.text_dataset import aggregate_daily_text_features


def _synthetic_ohlcv(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates)


def _synthetic_text_daily(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """A handful of real-shaped video records on a sparse subset of business days.

    Values vary per video (via a seeded RNG) rather than being identical -- identical
    values give zero variance, and rolling z-scoring a zero-variance column is 0/0 = NaN
    by construction, which an earlier version of this fixture triggered, making every row
    look "unavailable" for reasons unrelated to the join/masking logic under test."""
    rng = np.random.default_rng(3)
    records = [
        {
            "video_id": f"v{i}", "title": "t", "published_at": d.isoformat() + "Z",
            "sentiment_polarity": float(rng.uniform(-0.5, 0.5)),
            "hype_score": float(rng.uniform(0.1, 0.9)),
            "intensity": float(rng.uniform(0.2, 0.8)),
            "role": "unclassified",
            **{f"emotion_{c}": float(rng.uniform(0.0, 0.5)) for c in
               ["confidence", "uncertainty", "optimism", "fear", "caution", "assertiveness", "evasiveness"]},
        }
        for i, d in enumerate(dates)
    ]
    return aggregate_daily_text_features(records)


def test_dataset_without_text_has_no_text_columns():
    df = _synthetic_ohlcv()
    bundle = build_daily_dataset_from_ohlcv(df, "SYN", include_microstructure=False)
    assert bundle.text_columns == []
    assert "text_available" not in bundle.features.columns


def test_dataset_with_empty_text_daily_gets_zero_availability_mask():
    df = _synthetic_ohlcv()
    empty_text = aggregate_daily_text_features([])  # caller asked for text, none found
    bundle = build_daily_dataset_from_ohlcv(df, "SYN", include_microstructure=False, text_daily=empty_text)
    assert bundle.text_columns == ["text_available"]
    assert (bundle.features["text_available"] == 0.0).all()


def test_dataset_with_real_shaped_text_daily_sets_availability_and_zscores():
    df = _synthetic_ohlcv()
    # pick 15 sparse dates spread across the series that actually exist in df's index
    video_dates = df.index[::20][:15]
    text_daily = _synthetic_text_daily(video_dates)

    bundle = build_daily_dataset_from_ohlcv(df, "SYN", include_microstructure=False, text_daily=text_daily)

    assert "text_available" in bundle.text_columns
    assert "text_sentiment_polarity" in bundle.text_columns
    assert "text_n_videos" in bundle.text_columns

    # at least SOME rows must be marked available (the video dates, once enough history
    # has accumulated for the rolling z-score warm-up)
    assert bundle.features["text_available"].sum() > 0
    # rows with no video must be exactly 0, never a fabricated nonzero value
    unavailable = bundle.features[bundle.features["text_available"] == 0.0]
    assert (unavailable["text_sentiment_polarity"] == 0.0).all()


def test_text_columns_included_in_features_dtype_numeric():
    df = _synthetic_ohlcv()
    video_dates = df.index[::15][:20]
    text_daily = _synthetic_text_daily(video_dates)
    bundle = build_daily_dataset_from_ohlcv(df, "SYN", include_microstructure=False, text_daily=text_daily)
    for col in bundle.text_columns:
        assert pd.api.types.is_numeric_dtype(bundle.features[col])
        assert not bundle.features[col].isna().any()
