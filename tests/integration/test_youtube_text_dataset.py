"""Integration test for the real YouTube -> ASEM -> daily-text-features pipeline.

Deliberately tiny scope (one narrow window, one video) to avoid contributing to the
IP-blocking behavior observed in scripts/fetch_youtube_text_dataset.py's real run (23/35
transcript requests blocked). Skips gracefully if YouTube blocks this attempt too, since
that is a real external condition, not a defect in this code -- see docs/LIMITATIONS.md.
"""
from datetime import datetime, timedelta, timezone

import pytest

from emafrs.nlp.text_dataset import (
    aggregate_daily_text_features,
    fetch_channel_video_affective_records,
    fetch_video_affective_records,
)

pytestmark = pytest.mark.integration


def test_fetch_and_aggregate_real_youtube_videos():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=14)

    records, stats = fetch_video_affective_records(
        "Federal Reserve", start, end, window_days=14, max_results_per_window=1, max_chars=500,
        transcript_request_delay_seconds=0.5,
    )

    assert stats.n_windows_searched == 1
    assert stats.youtube_quota_units_used == 100

    if stats.n_videos_processed == 0:
        pytest.skip(
            f"No video was successfully processed in this run (found={stats.n_videos_found}, "
            f"transcripts_available={stats.n_transcripts_available}) -- likely YouTube's "
            "undocumented IP-blocking of transcript requests, a real external condition, "
            "not a defect in this code. See docs/LIMITATIONS.md."
        )

    assert len(records) == stats.n_videos_processed
    for r in records:
        assert -1.0 <= r["sentiment_polarity"] <= 1.0
        assert 0.0 <= r["hype_score"] <= 1.0
        assert r["role"] in ("unclassified",)  # no verified-channel allowlist entries configured

    daily = aggregate_daily_text_features(records)
    assert len(daily) <= len(records)  # same-day videos collapse to one row
    assert daily["text_n_items"].sum() == len(records)


def test_known_financial_news_channel_ids_are_still_correct():
    """Regression guard, real API call: re-resolves the real handles behind
    KNOWN_FINANCIAL_NEWS_CHANNELS's hardcoded IDs and asserts they still match. Catches
    silent drift (a channel ID getting reassigned/deleted) rather than trusting a
    hardcoded value forever -- see youtube_client.py's comment on '@BloombergTelevision'
    resolving to an unrelated 89-subscriber channel, which is exactly the failure mode
    this test exists to catch going forward."""
    from emafrs.data.youtube_client import KNOWN_FINANCIAL_NEWS_CHANNELS, YouTubeClient

    client = YouTubeClient()
    live_ids = {
        "CNBC": client.resolve_channel_id("@CNBC"),
        "Bloomberg Television": client.resolve_channel_id("@markets"),
    }
    for name, hardcoded_id in KNOWN_FINANCIAL_NEWS_CHANNELS.items():
        assert live_ids[name] == hardcoded_id, f"{name}'s hardcoded channel ID no longer matches its real handle"


def test_channel_targeted_fetch_finds_real_recent_cnbc_bloomberg_videos():
    """Real, publisher-targeted discovery (get_recent_uploads, 1 quota unit/channel) --
    proves this actually finds real recent videos from CNBC's/Bloomberg's live-verified
    channel IDs, distinct from fetch_video_affective_records's generic keyword search.
    Transcript IP-blocking (docs/LIMITATIONS.md) is a separate, already-documented
    problem that channel targeting does not fix -- this test only asserts that real
    videos were FOUND for these specific channels, and skips gracefully (rather than
    failing) if none of them happened to have a retrievable transcript right now."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=5)

    records, stats = fetch_channel_video_affective_records(
        start, end, max_results_per_channel=3, max_chars=500, transcript_request_delay_seconds=0.5
    )

    assert stats.n_videos_found > 0, "Expected at least one real recent CNBC/Bloomberg upload in the last 5 days"

    if stats.n_videos_processed == 0:
        pytest.skip(
            f"Found {stats.n_videos_found} real CNBC/Bloomberg videos but none had a retrievable "
            "transcript in this run -- YouTube's IP-blocking, a real external condition, not a "
            "defect in channel targeting. See docs/LIMITATIONS.md."
        )

    assert len(records) == stats.n_videos_processed
    for r in records:
        assert r["source"] == "youtube"
        assert -1.0 <= r["sentiment_polarity"] <= 1.0

    daily = aggregate_daily_text_features(records)
    assert daily["text_n_items"].sum() == len(records)
