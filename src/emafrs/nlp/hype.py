"""Emotional hype score and hype acceleration -- PDF Section 6.4, Fig. 2.

S_t = C_s * (w1*e_text + w2*e_audio + w3*e_video). Only the text modality is populated by
the live pipeline (see docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md and
data/youtube_client.py's modality-honesty note) -- w_audio and w_video are 0 unless a
case-study video is specifically processed for real audio/video features, which is out of
scope for the live/production pipeline.
"""
from __future__ import annotations

import pandas as pd

from emafrs.nlp.emotion import EmotionScores

# Hype-relevant emotions: high optimism/assertiveness/confidence with low caution/uncertainty
# is the textbook signature of promotional/hype language (PDF Sec 3.1-3.2 background).
_HYPE_POSITIVE = ("optimism", "assertiveness", "confidence")
_HYPE_NEGATIVE = ("caution", "uncertainty", "evasiveness")


def text_hype_component(emotion_scores: EmotionScores) -> float:
    """e_text in PDF's S_t formula: mean of hype-positive emotion scores minus mean of
    hype-negative emotion scores, clipped to [0, 1]."""
    pos = sum(emotion_scores.scores[c] for c in _HYPE_POSITIVE) / len(_HYPE_POSITIVE)
    neg = sum(emotion_scores.scores[c] for c in _HYPE_NEGATIVE) / len(_HYPE_NEGATIVE)
    return max(0.0, min(1.0, pos - neg + 0.5))  # recenter so neutral input ~0.5, not 0


def compute_hype_score(
    credibility: float,
    emotion_scores: EmotionScores,
    audio_component: float | None = None,
    video_component: float | None = None,
    w_text: float = 1.0,
    w_audio: float = 0.0,
    w_video: float = 0.0,
) -> float:
    """S_t = C_s * (w1*e_text + w2*e_audio + w3*e_video). Defaults to text-only (w_audio=
    w_video=0) since audio/video components are None in the live pipeline; passing a real
    audio/video component (case-study path) requires the caller to also set nonzero weights."""
    e_text = text_hype_component(emotion_scores)
    total = w_text * e_text
    weight_sum = w_text
    if audio_component is not None and w_audio > 0:
        total += w_audio * audio_component
        weight_sum += w_audio
    if video_component is not None and w_video > 0:
        total += w_video * video_component
        weight_sum += w_video
    normalized = total / weight_sum if weight_sum > 0 else 0.0
    return credibility * normalized


def hype_acceleration(hype_scores: pd.Series) -> pd.Series:
    """H_t = dS_t/dt (PDF Sec 6.4, Fig. 2) -- first difference of the hype score series."""
    return hype_scores.diff()


def spike_alert(hype_acceleration_series: pd.Series, threshold: float) -> pd.Series:
    """PDF Claim 18: H_t > tau_H triggers a Spike Alert independently of the full risk model."""
    return hype_acceleration_series > threshold
