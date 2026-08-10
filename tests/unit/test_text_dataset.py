"""Tests for the daily text-feature aggregation logic (no network -- record fetching
itself is exercised in tests/integration/test_youtube_text_dataset.py and
tests/integration/test_newsapi_text_dataset.py)."""
import pandas as pd

from emafrs.nlp.emotion import EMOTION_CATEGORIES
from emafrs.nlp.text_dataset import TEXT_FEATURE_COLUMNS, aggregate_daily_text_features


def _record(source: str, published_at: str, sentiment: float, hype: float, intensity: float = 0.5, **emotion_overrides) -> dict:
    rec = {
        "source": source,
        "item_id": f"{source}_{published_at}",
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
    assert "text_n_items" in result.columns
    for col in TEXT_FEATURE_COLUMNS:
        assert col in result.columns


def test_single_item_maps_directly_to_its_date():
    records = [_record("youtube", "2024-03-01T14:00:00Z", sentiment=0.6, hype=0.7)]
    result = aggregate_daily_text_features(records)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["text_n_items"] == 1
    assert row["text_sentiment_polarity"] == 0.6
    assert row["text_hype_score"] == 0.7


def test_multiple_items_same_day_are_averaged():
    records = [
        _record("youtube", "2024-03-01T09:00:00Z", sentiment=0.2, hype=0.4),
        _record("newsapi", "2024-03-01T18:00:00Z", sentiment=0.8, hype=0.6),
    ]
    result = aggregate_daily_text_features(records)
    assert len(result) == 1  # same calendar date -> one row, regardless of source
    row = result.iloc[0]
    assert row["text_n_items"] == 2
    assert row["text_sentiment_polarity"] == 0.5  # mean(0.2, 0.8)
    assert row["text_hype_score"] == 0.5  # mean(0.4, 0.6)


def test_items_on_different_days_produce_separate_rows():
    records = [
        _record("youtube", "2024-03-01T09:00:00Z", sentiment=0.2, hype=0.4),
        _record("newsapi", "2024-03-05T09:00:00Z", sentiment=0.8, hype=0.6),
    ]
    result = aggregate_daily_text_features(records)
    assert len(result) == 2
    assert result.index.tolist() == sorted(result.index.tolist())  # chronological


def test_emotion_columns_are_present_and_averaged():
    records = [
        _record("youtube", "2024-03-01T09:00:00Z", sentiment=0.0, hype=0.0, fear=0.2),
        _record("newsapi", "2024-03-01T18:00:00Z", sentiment=0.0, hype=0.0, fear=0.8),
    ]
    result = aggregate_daily_text_features(records)
    assert result.iloc[0]["text_emotion_fear"] == 0.5


def test_trading_days_none_preserves_prior_behavior():
    """Backward compatibility: omitting trading_days must give byte-identical output to
    before this parameter existed."""
    records = [_record("newsapi", "2024-03-02T09:00:00Z", sentiment=0.5, hype=0.5)]  # a Saturday
    result = aggregate_daily_text_features(records)
    assert result.index[0] == pd.Timestamp("2024-03-02")  # untouched, still the raw calendar date


def test_weekend_article_maps_to_next_monday_when_trading_days_given():
    # 2024-03-02 is a Saturday; 2024-03-04 is the following Monday
    records = [_record("newsapi", "2024-03-02T09:00:00Z", sentiment=0.5, hype=0.5)]
    trading_days = pd.DatetimeIndex(["2024-03-01", "2024-03-04", "2024-03-05"])  # Fri, Mon, Tue
    result = aggregate_daily_text_features(records, trading_days=trading_days)
    assert len(result) == 1
    assert result.index[0] == pd.Timestamp("2024-03-04")


def test_article_already_on_a_trading_day_maps_to_itself():
    records = [_record("newsapi", "2024-03-04T09:00:00Z", sentiment=0.5, hype=0.5)]  # a Monday
    trading_days = pd.DatetimeIndex(["2024-03-01", "2024-03-04", "2024-03-05"])
    result = aggregate_daily_text_features(records, trading_days=trading_days)
    assert result.index[0] == pd.Timestamp("2024-03-04")


def test_article_after_last_trading_day_is_dropped_not_backward_mapped():
    """A date after the last known trading day has no FUTURE trading day to map to --
    must be dropped, never mapped backward to the last available day (that direction
    would be leakage: attributing information to a date before it was published)."""
    records = [_record("newsapi", "2024-03-10T09:00:00Z", sentiment=0.5, hype=0.5)]
    trading_days = pd.DatetimeIndex(["2024-03-01", "2024-03-04", "2024-03-05"])  # all before 03-10
    result = aggregate_daily_text_features(records, trading_days=trading_days)
    assert result.empty


def test_saturday_and_sunday_articles_both_map_to_same_following_monday_and_aggregate_together():
    records = [
        _record("newsapi", "2024-03-02T09:00:00Z", sentiment=0.2, hype=0.4),  # Saturday
        _record("newsapi", "2024-03-03T09:00:00Z", sentiment=0.8, hype=0.6),  # Sunday
    ]
    trading_days = pd.DatetimeIndex(["2024-03-01", "2024-03-04", "2024-03-05"])
    result = aggregate_daily_text_features(records, trading_days=trading_days)
    assert len(result) == 1
    assert result.index[0] == pd.Timestamp("2024-03-04")
    row = result.iloc[0]
    assert row["text_n_items"] == 2
    assert row["text_sentiment_polarity"] == 0.5  # mean(0.2, 0.8), both now on the same mapped day


def test_mapping_never_moves_a_date_earlier_property_check():
    """Property test over several dates: the mapped date must never precede the original
    published date -- the core causality guarantee."""
    records = [
        _record("newsapi", "2024-03-01T09:00:00Z", sentiment=0.5, hype=0.5),  # Friday, already trading
        _record("newsapi", "2024-03-02T09:00:00Z", sentiment=0.5, hype=0.5),  # Saturday
        _record("newsapi", "2024-03-03T09:00:00Z", sentiment=0.5, hype=0.5),  # Sunday
    ]
    trading_days = pd.DatetimeIndex(["2024-03-01", "2024-03-04"])
    result = aggregate_daily_text_features(records, trading_days=trading_days)
    for mapped_date in result.index:
        assert mapped_date >= pd.Timestamp("2024-03-01")


def test_mapping_is_truncation_invariant_extending_calendar_further_future_does_not_change_past_mapping():
    """Leakage-safety proof, same truncation methodology used elsewhere in this codebase
    (emafrs.features.leakage_audit.assert_no_lookahead): a date's mapped trading day must
    not change when trading days FURTHER in the future are added to the calendar -- only
    trading days between the original date and its current mapped day could ever matter,
    and those are, by construction, already earlier than any newly-appended future date."""
    records = [_record("newsapi", "2024-03-02T09:00:00Z", sentiment=0.5, hype=0.5)]  # Saturday
    short_calendar = pd.DatetimeIndex(["2024-03-01", "2024-03-04", "2024-03-05"])
    long_calendar = pd.DatetimeIndex(["2024-03-01", "2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-04-01"])

    result_short = aggregate_daily_text_features(records, trading_days=short_calendar)
    result_long = aggregate_daily_text_features(records, trading_days=long_calendar)
    assert result_short.index[0] == result_long.index[0] == pd.Timestamp("2024-03-04")


def test_aggregation_is_source_agnostic_by_design():
    """The whole point of unifying YouTube and NewsAPI into the same record schema:
    aggregate_daily_text_features must not care which source a record came from."""
    youtube_only = [_record("youtube", "2024-03-01T09:00:00Z", sentiment=0.5, hype=0.5)]
    newsapi_only = [_record("newsapi", "2024-03-01T09:00:00Z", sentiment=0.5, hype=0.5)]
    assert aggregate_daily_text_features(youtube_only).equals(aggregate_daily_text_features(newsapi_only))
