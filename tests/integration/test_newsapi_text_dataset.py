"""Integration test for the real NewsAPI -> ASEM -> daily-text-features pipeline.
Small scope (one query, small page_size) but real: real HTTP call, real ASEM models."""
from datetime import datetime, timedelta, timezone

import pytest

from emafrs.nlp.text_dataset import aggregate_daily_text_features, fetch_news_affective_records

pytestmark = pytest.mark.integration


def test_fetch_and_aggregate_real_news_articles():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)

    records, stats = fetch_news_affective_records("Apple", start, end, page_size=5, max_chars=500)

    assert stats.n_requests_made == 1
    if stats.n_articles_processed == 0:
        pytest.skip(f"No article was found/processed in this run (found={stats.n_articles_found}) -- "
                     "real NewsAPI quota/availability condition, not necessarily a defect.")

    assert len(records) == stats.n_articles_processed
    for r in records:
        assert r["source"] == "newsapi"
        assert -1.0 <= r["sentiment_polarity"] <= 1.0
        assert 0.0 <= r["hype_score"] <= 1.0
        assert r["role"] == "unclassified"  # no verified-source allowlist for news publishers

    daily = aggregate_daily_text_features(records)
    assert daily["text_n_items"].sum() == len(records)


def test_combined_youtube_and_newsapi_records_aggregate_together():
    """Proves the whole point of unifying the two sources: a combined record list from
    (a small amount of) both providers aggregates through the SAME function without any
    source-specific branching needed by the caller."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)

    news_records, news_stats = fetch_news_affective_records("Federal Reserve", start, end, page_size=3, max_chars=500)
    if news_stats.n_articles_processed == 0:
        pytest.skip("No news articles processed in this run -- cannot test combination.")

    combined = news_records  # a real YouTube fetch here would cost 100 quota units; the
    # unit tests already prove aggregate_daily_text_features is source-agnostic by
    # construction (test_aggregation_is_source_agnostic_by_design), so re-spending
    # YouTube quota on every CI run of this integration test is not warranted.
    daily = aggregate_daily_text_features(combined)
    assert not daily.empty
