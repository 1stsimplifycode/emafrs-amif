"""Builds a daily-aggregated TEXT/affective feature dataset from real financial
communication sources -- YouTube video transcripts and NewsAPI articles -- the text
modality dataset in this reconstruction, closing part of the gap documented in
docs/ABLATION_STUDY.md rows B/C/D/E/G/H.

Design (each choice disclosed, not silent):

1. YouTube search.list costs 100 quota units/call (10,000/day budget) -- the date range is
   chunked into windows (default 30 days) so a multi-year backfill doesn't require one
   call per day. `max_results_per_window` caps videos considered per window.
2. NewsAPI's free "Developer" tier restricts `/v2/everything` to roughly the trailing
   month regardless of the `start_date` requested -- this is a FUNDAMENTALLY DIFFERENT
   coverage shape than YouTube: dense but recent-only, vs. YouTube's sparse-but-
   potentially-historical (subject to its own IP-blocking limitation). A caller building a
   historical training dataset must understand that NewsAPI-derived rows will cluster at
   the very end of the timeline (likely inside the test split, never the training split)
   -- see docs/TEXT_MODALITY_PILOT.md for what this means in practice, discovered by
   actually running it, not assumed in advance.
3. Each item's affective feature vector comes from real ASEM processing (FinBERT
   sentiment + zero-shot emotion + role/credibility/hype) over: for YouTube, the video
   transcript truncated to `max_chars`; for NewsAPI, `title + ". " + description`
   (NewsAPI's free tier truncates `content` to ~200 chars with a paywall notice appended,
   so `description`, a real un-truncated summary field, is used instead of the truncated
   `content` field).
4. Multiple items on the same calendar date (from either or both sources) are aggregated
   by MEAN (sentiment polarity, hype score, each of the 7 emotion scores, intensity) plus
   an item COUNT (`text_n_items` -- renamed from an earlier `text_n_videos` now that
   non-video sources feed the same aggregation). The 384-dim semantic embedding is
   computed per-item by ASEM but NOT included in this tabular daily dataset, to keep the
   text modality's dimensionality comparable to the other modalities.
5. Dates with no item get `text_available=0` and zero-filled feature columns -- the same
   masking pattern already used for the microstructure modality in emafrs.models.dataset.
6. No project-specific relevance filtering beyond each provider's own search relevance
   ranking for the query -- a real, disclosed limitation.
7. Role/credibility default to "unclassified"/authority-tier-0.4 for every item unless a
   YouTube channel_id happens to be in the verified allowlist (empty by default) -- never
   inferred from a channel/article/publisher name. Every record carries a `source` field
   ("youtube" or "newsapi") for provenance.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from emafrs.data.newsapi_client import NewsAPIClient
from emafrs.data.youtube_client import YouTubeClient
from emafrs.nlp.asem import AffectiveSignalExtractionModule
from emafrs.nlp.emotion import EMOTION_CATEGORIES

logger = logging.getLogger("emafrs.nlp.text_dataset")

TEXT_FEATURE_COLUMNS = (
    ["text_sentiment_polarity", "text_hype_score", "text_intensity"]
    + [f"text_emotion_{c}" for c in EMOTION_CATEGORIES]
)


@dataclass
class VideoProcessingStats:
    n_windows_searched: int
    n_videos_found: int
    n_transcripts_available: int
    n_videos_processed: int
    youtube_quota_units_used: int  # 100 per search.list call -- this function's usage only


@dataclass
class NewsProcessingStats:
    n_requests_made: int
    n_articles_found: int
    n_articles_processed: int


def _chunk_date_range(start: datetime, end: datetime, window_days: int):
    cur = start
    while cur < end:
        window_end = min(cur + timedelta(days=window_days), end)
        yield cur, window_end
        cur = window_end


def _process_text_with_asem(
    asem: AffectiveSignalExtractionModule, text: str, provider: str, entity: str,
    timestamp: datetime, metadata: dict,
) -> dict | None:
    """Shared ASEM-processing-into-record-dict step for both sources."""
    try:
        feature_vec = asem.process(text=text, provider=provider, entity=entity, timestamp=timestamp, metadata=metadata)
    except Exception as exc:
        logger.warning("ASEM processing failed for %s (%s): %s", provider, metadata, exc)
        return None
    return {
        "sentiment_polarity": feature_vec.sentiment.polarity,
        "hype_score": feature_vec.hype_score,
        "intensity": feature_vec.intensity,
        "role": feature_vec.role,
        **{f"emotion_{c}": feature_vec.emotion.scores[c] for c in EMOTION_CATEGORIES},
    }


def fetch_video_affective_records(
    query: str,
    start_date: datetime,
    end_date: datetime,
    window_days: int = 30,
    max_results_per_window: int = 5,
    max_chars: int = 2000,
    asem: AffectiveSignalExtractionModule | None = None,
    transcript_request_delay_seconds: float = 1.5,
) -> tuple[list[dict], VideoProcessingStats]:
    """Searches YouTube for `query` across [start_date, end_date] in `window_days` chunks,
    fetches each video's transcript, and runs it through ASEM. Returns (records, stats);
    stats reports exactly what was searched/found/processed and the real YouTube quota
    cost of THIS call, so a caller can decide whether to run it again without guessing.

    REAL LIMITATION OBSERVED IN USE: youtube-transcript-api's underlying (undocumented,
    unofficial) endpoint IP-blocks callers that request many transcripts in quick
    succession -- in one real run from this development environment, 23 of 35 found
    videos failed with "YouTube is blocking requests from your IP." `transcript_request_
    delay_seconds` adds a courtesy pause between transcript requests to reduce (not
    eliminate -- this is an undocumented anti-bot heuristic, not a published rate limit)
    the chance of this. See docs/LIMITATIONS.md; do not interpret a failed transcript
    fetch as "no transcript exists" without checking the actual exception."""
    client = YouTubeClient()
    asem = asem or AffectiveSignalExtractionModule()

    records: list[dict] = []
    n_windows = 0
    n_found = 0
    n_with_transcript = 0

    for window_start, window_end in _chunk_date_range(start_date, end_date, window_days):
        n_windows += 1
        try:
            items, _ = client.search_videos(query, window_start, window_end, max_results=max_results_per_window)
        except Exception as exc:
            logger.warning("YouTube search failed for window %s-%s: %s", window_start, window_end, exc)
            continue
        n_found += len(items)

        for item in items:
            try:
                text, _ = client.get_transcript_text(item["video_id"])
            except Exception as exc:
                logger.warning("Transcript fetch failed for video_id=%s: %s", item["video_id"], exc)
                continue
            finally:
                if transcript_request_delay_seconds > 0:
                    time.sleep(transcript_request_delay_seconds)
            if text is None:
                continue
            n_with_transcript += 1
            text = text[:max_chars]

            published_at = datetime.fromisoformat(item["published_at"].replace("Z", "+00:00"))
            fields = _process_text_with_asem(
                asem, text, "youtube", query, published_at,
                {"channel_id": item["channel_id"], "video_id": item["video_id"]},
            )
            if fields is None:
                continue

            records.append({
                "source": "youtube",
                "item_id": item["video_id"],
                "title": item["title"],
                "published_at": item["published_at"],
                **fields,
            })

    stats = VideoProcessingStats(
        n_windows_searched=n_windows, n_videos_found=n_found,
        n_transcripts_available=n_with_transcript, n_videos_processed=len(records),
        youtube_quota_units_used=n_windows * 100,
    )
    return records, stats


def fetch_news_affective_records(
    query: str,
    start_date: datetime,
    end_date: datetime,
    page_size: int = 50,
    max_chars: int = 1000,
    asem: AffectiveSignalExtractionModule | None = None,
) -> tuple[list[dict], NewsProcessingStats]:
    """Searches NewsAPI for `query` across [start_date, end_date] (a SINGLE call -- no
    chunking needed since the free tier's own ~1-month lookback cap makes multi-window
    chunking pointless) and runs each article through ASEM.

    Uses `title + ". " + description` as the ASEM input, NOT `content` -- NewsAPI's free
    tier truncates `content` to ~200 characters with a "[+N chars]" paywall notice
    appended, which is not real article text; `description` is a real, untruncated
    provider-written summary field."""
    client = NewsAPIClient()
    asem = asem or AffectiveSignalExtractionModule()

    articles, _ = client.search(query, start_date.date(), end_date.date(), page_size=page_size)

    records: list[dict] = []
    for article in articles:
        title = article.get("title") or ""
        description = article.get("description") or ""
        text = f"{title}. {description}".strip()
        if not text or text == ".":
            continue
        text = text[:max_chars]

        published_at_raw = article.get("publishedAt")
        if not published_at_raw:
            continue
        try:
            published_at = datetime.fromisoformat(published_at_raw.replace("Z", "+00:00"))
        except ValueError:
            continue

        fields = _process_text_with_asem(
            asem, text, "newsapi", query, published_at,
            {"source_name": (article.get("source") or {}).get("name"), "url": article.get("url")},
        )
        if fields is None:
            continue

        records.append({
            "source": "newsapi",
            "item_id": article.get("url"),
            "title": title,
            "published_at": published_at_raw,
            **fields,
        })

    stats = NewsProcessingStats(
        n_requests_made=1, n_articles_found=len(articles), n_articles_processed=len(records),
    )
    return records, stats


def aggregate_daily_text_features(records: list[dict]) -> pd.DataFrame:
    """Aggregates per-item (video or article, from either/both sources) affective records
    to one row per calendar date (mean across same-day items + an item count). Returns an
    EMPTY (zero-row) DataFrame with the correct columns if `records` is empty, never an
    error -- callers join this against the main dataset and must treat missing dates as
    text_available=0, not a failure."""
    if not records:
        return pd.DataFrame(columns=["text_n_items", *TEXT_FEATURE_COLUMNS])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["published_at"], utc=True).dt.tz_localize(None).dt.normalize()

    value_cols = ["sentiment_polarity", "hype_score", "intensity"] + [f"emotion_{c}" for c in EMOTION_CATEGORIES]
    daily = df.groupby("date")[value_cols].mean()
    daily["text_n_items"] = df.groupby("date").size()

    daily = daily.rename(columns={
        "sentiment_polarity": "text_sentiment_polarity",
        "hype_score": "text_hype_score",
        "intensity": "text_intensity",
        **{f"emotion_{c}": f"text_emotion_{c}" for c in EMOTION_CATEGORIES},
    })
    return daily[["text_n_items", *TEXT_FEATURE_COLUMNS]]
