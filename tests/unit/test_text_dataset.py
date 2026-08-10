"""Tests for the daily text-feature aggregation logic (no network -- record fetching
itself is exercised in tests/integration/test_youtube_text_dataset.py)."""
import pandas as pd

from emafrs.nlp.emotion import EMOTION_CATEGORIES
from emafrs.nlp.text_dataset import TEXT_FEATURE_COLUMNS, aggregate_daily_text_features


def _record(published_at: str, sentiment: float, hype: float, intensity: float = 0.5, **emotion_overrides) -> dict:
    rec = {
        "video_id": f"v_{published_at}",
        "title": "t",
        "published_at": published_at,
        "sentiment_polarity": sentiment,
        "hype_score": hype,
        "intensity": intensity,
        "role": "unclassified",
    }
    for c in EMOTION_CATEGORIES:
        rec[f"emotion_{c}"] = emotion_overrides.get(c, 0.1)
    return rec


def test_empty_records_returns_empty_frame_with_correct_columns():
    result = aggregate_daily_text_features([])
    assert result.empty
    assert "text_n_videos" in result.columns
    for col in TEXT_FEATURE_COLUMNS:
        assert col in result.columns


def test_single_video_maps_directly_to_its_date():
    records = [_record("2024-03-01T14:00:00Z", sentiment=0.6, hype=0.7)]
    result = aggregate_daily_text_features(records)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["text_n_videos"] == 1
    assert row["text_sentiment_polarity"] == 0.6
    assert row["text_hype_score"] == 0.7


def test_multiple_videos_same_day_are_averaged():
    records = [
        _record("2024-03-01T09:00:00Z", sentiment=0.2, hype=0.4),
        _record("2024-03-01T18:00:00Z", sentiment=0.8, hype=0.6),
    ]
    result = aggregate_daily_text_features(records)
    assert len(result) == 1  # same calendar date -> one row
    row = result.iloc[0]
    assert row["text_n_videos"] == 2
    assert row["text_sentiment_polarity"] == 0.5  # mean(0.2, 0.8)
    assert row["text_hype_score"] == 0.5  # mean(0.4, 0.6)


def test_videos_on_different_days_produce_separate_rows():
    records = [
        _record("2024-03-01T09:00:00Z", sentiment=0.2, hype=0.4),
        _record("2024-03-05T09:00:00Z", sentiment=0.8, hype=0.6),
    ]
    result = aggregate_daily_text_features(records)
    assert len(result) == 2
    assert result.index.tolist() == sorted(result.index.tolist())  # chronological


def test_emotion_columns_are_present_and_averaged():
    records = [
        _record("2024-03-01T09:00:00Z", sentiment=0.0, hype=0.0, fear=0.2),
        _record("2024-03-01T18:00:00Z", sentiment=0.0, hype=0.0, fear=0.8),
    ]
    result = aggregate_daily_text_features(records)
    assert result.iloc[0]["text_emotion_fear"] == 0.5
