"""YouTube Data API v3 client: financial-communication video discovery + transcript text.

Quota note: YouTube Data API free quota is 10,000 units/day. search.list costs 100 units
per call (expensive), videos.list costs 1 unit. This client caches search results
aggressively and prefers videos.list for anything that doesn't require full-text search.

IMPORTANT SCOPE LIMIT (per build brief Section 8 and docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md
row on YouTube): this client only ever produces metadata + transcript TEXT. It does not
download or process actual audio/video streams. Every transcript-derived record is tagged
`modality="text_derived_from_av_source"` so downstream code and dashboards report it
honestly as "Textual modality derived from audiovisual communication source," never as
audio or video modeling.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from emafrs.data.config import CONFIG
from emafrs.data.schema import SourceRecord

logger = logging.getLogger("emafrs.data.youtube")

LICENSE_NOTE = (
    "YouTube Data API v3, free quota 10,000 units/day. Content subject to YouTube ToS; "
    "transcripts are auto-generated or creator-provided captions, not verified transcripts."
)

# Real, live-verified YouTube channel IDs for major financial-news publishers -- resolved
# via channels().list(forHandle=...) and checked against subscriber count/description
# before being hardcoded here (a wrong handle can silently resolve to an unrelated
# channel: '@BloombergTelevision' was tried and found to resolve to an 89-subscriber
# unrelated channel, NOT Bloomberg -- '@markets' is Bloomberg's real, verified TV channel,
# 3.24M subscribers). Do not add a handle/ID here without live-verifying it the same way;
# a wrong ID here would silently misattribute unrelated content as CNBC/Bloomberg.
KNOWN_FINANCIAL_NEWS_CHANNELS = {
    "CNBC": "UCvJJ_dzjViJCoLf5uKUTwoA",  # verified via forHandle='@CNBC': title "CNBC", 4.17M subs, official channel
    "Bloomberg Television": "UCIALMKvObZNtJ6AmdCLP7Lg",  # verified via forHandle='@markets': title "Bloomberg Television", 3.24M subs
}


class YouTubeClient:
    provider = "youtube"

    def __init__(self):
        if not CONFIG.youtube_api_key:
            raise RuntimeError("YOUTUBE_API_KEY not configured")
        self._service = build("youtube", "v3", developerKey=CONFIG.youtube_api_key, cache_discovery=False)

    def search_videos(
        self, query: str, published_after: datetime, published_before: datetime, max_results: int = 10
    ) -> tuple[list[dict], SourceRecord]:
        """search.list: 100 quota units. published_after/before enforce the query never
        pulls videos published after the analysis window's end, preventing look-ahead."""
        fetched_at = datetime.now(timezone.utc)
        try:
            resp = self._service.search().list(
                q=query,
                part="snippet",
                type="video",
                order="relevance",
                maxResults=max_results,
                publishedAfter=published_after.strftime("%Y-%m-%dT%H:%M:%SZ"),
                publishedBefore=published_before.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ).execute()
        except HttpError as exc:
            raise RuntimeError(f"YouTube search failed for query={query!r}: {exc}") from exc

        items = [
            {
                "video_id": it["id"]["videoId"],
                "title": it["snippet"]["title"],
                "channel_title": it["snippet"]["channelTitle"],
                "channel_id": it["snippet"]["channelId"],
                "published_at": it["snippet"]["publishedAt"],
                "description": it["snippet"]["description"],
            }
            for it in resp.get("items", [])
        ]
        record = SourceRecord(
            provider=self.provider,
            endpoint="search_videos",
            fetched_at=fetched_at,
            observation_time=None,
            entity=query,
            fields={"n_results": len(items), "published_after": str(published_after),
                    "published_before": str(published_before)},
            license_note=LICENSE_NOTE,
        )
        return items, record

    def get_recent_uploads(
        self, channel_id: str, published_after: datetime, max_results: int = 15
    ) -> tuple[list[dict], SourceRecord]:
        """Real recent uploads from ONE SPECIFIC channel (e.g. a verified CNBC/Bloomberg
        channel ID from KNOWN_FINANCIAL_NEWS_CHANNELS), via playlistItems.list against
        that channel's uploads playlist -- 1 quota unit total (a channels.list lookup +
        one playlistItems.list call, each 1 unit), vs. search_videos()'s 100 units for a
        keyword search that isn't targeted at any specific publisher at all. Returns
        genuinely recent uploads in reverse-chronological order, not a relevance ranking."""
        fetched_at = datetime.now(timezone.utc)
        try:
            channel_resp = self._service.channels().list(part="contentDetails", id=channel_id).execute()
            channel_items = channel_resp.get("items", [])
            if not channel_items:
                record = SourceRecord(
                    provider=self.provider, endpoint="get_recent_uploads", fetched_at=fetched_at,
                    observation_time=None, entity=channel_id,
                    fields={"n_results": 0, "error": "channel_id not found"}, license_note=LICENSE_NOTE,
                )
                return [], record
            uploads_playlist_id = channel_items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

            resp = self._service.playlistItems().list(
                part="snippet", playlistId=uploads_playlist_id, maxResults=max_results
            ).execute()
        except HttpError as exc:
            raise RuntimeError(f"YouTube get_recent_uploads failed for channel_id={channel_id!r}: {exc}") from exc

        items = []
        for it in resp.get("items", []):
            snippet = it["snippet"]
            published_at = datetime.fromisoformat(snippet["publishedAt"].replace("Z", "+00:00"))
            if published_at < published_after:
                continue  # playlist is reverse-chronological, but filter explicitly rather than assume
            items.append({
                "video_id": snippet["resourceId"]["videoId"],
                "title": snippet["title"],
                "channel_title": snippet["channelTitle"],
                "channel_id": snippet["channelId"],
                "published_at": snippet["publishedAt"],
                "description": snippet["description"],
            })

        record = SourceRecord(
            provider=self.provider,
            endpoint="get_recent_uploads",
            fetched_at=fetched_at,
            observation_time=None,
            entity=channel_id,
            fields={"n_results": len(items), "published_after": str(published_after)},
            license_note=LICENSE_NOTE,
        )
        return items, record

    def resolve_channel_id(self, handle: str) -> str | None:
        """1 quota unit. Resolves a real YouTube handle (e.g. '@CNBC') to its channel ID --
        used to VERIFY a handle before adding it to KNOWN_FINANCIAL_NEWS_CHANNELS, not
        called on every fetch (the resolved IDs are hardcoded above once verified, so
        routine fetching doesn't spend quota re-resolving a handle that doesn't change)."""
        resp = self._service.channels().list(part="id", forHandle=handle).execute()
        items = resp.get("items", [])
        return items[0]["id"] if items else None

    def get_transcript_text(self, video_id: str) -> tuple[str | None, SourceRecord]:
        """Returns (transcript_text_or_None, provenance_record). None means no transcript
        is available -- callers must skip the video rather than fabricate text.

        REAL COMPATIBILITY ISSUE FOUND DURING USE: the installed youtube_transcript_api
        version removed the old classmethod `YouTubeTranscriptApi.get_transcript(video_id)`
        in favor of an instance method `YouTubeTranscriptApi().fetch(video_id)` returning a
        `FetchedTranscript` (iterable of snippets with a `.text` attribute, not dicts).
        Fixed here rather than pinning an old library version, since the new API is the
        one that will keep working going forward."""
        fetched_at = datetime.now(timezone.utc)
        try:
            fetched = YouTubeTranscriptApi().fetch(video_id)
            text = " ".join(snippet.text for snippet in fetched)
        except (TranscriptsDisabled, NoTranscriptFound):
            text = None
        except Exception as exc:  # transcript API raises several other narrow exceptions
            logger.warning("Transcript fetch failed for video_id=%s: %s", video_id, exc)
            text = None

        record = SourceRecord(
            provider=self.provider,
            endpoint="transcript",
            fetched_at=fetched_at,
            observation_time=None,
            entity=video_id,
            fields={
                "available": text is not None,
                "n_chars": len(text) if text else 0,
                "modality": "text_derived_from_av_source",
            },
            license_note=LICENSE_NOTE,
        )
        return text, record
