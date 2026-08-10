"""Integration test for the real YouTube -> ASEM -> daily-text-features pipeline.

Deliberately tiny scope (one narrow window, one video) to avoid contributing to the
IP-blocking behavior observed in scripts/fetch_youtube_text_dataset.py's real run (23/35
transcript requests blocked). Skips gracefully if YouTube blocks this attempt too, since
that is a real external condition, not a defect in this code -- see docs/LIMITATIONS.md.
"""
from datetime import datetime, timedelta, timezone

import pytest

from emafrs.nlp.text_dataset import aggregate_daily_text_features, fetch_video_affective_records

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
