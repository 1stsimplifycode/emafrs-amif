"""Production Sentiment Polarity Classifier (SPC) -- PDF Section 6.4.3.

Uses the financial-domain model (FinBERT) selected by the benchmark in
sentiment_benchmark.py / configs/nlp_models.yaml `sentiment_benchmark.financial_domain` --
see docs/NLP_MODEL_BENCHMARK.md for why this model was chosen over the generic and
other-domain alternatives, rather than picking it without justification.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SentimentResult:
    polarity: float  # s_t in [-1, +1]: P(positive) - P(negative)
    p_negative: float
    p_neutral: float
    p_positive: float


class SentimentPolarityClassifier:
    def __init__(self, hf_id: str = "ProsusAI/finbert"):
        from transformers import pipeline

        self.hf_id = hf_id
        self._pipe = pipeline("text-classification", model=hf_id, top_k=None)

    def score(self, text: str) -> SentimentResult:
        raw = self._pipe(text, truncation=True)[0]
        probs = {d["label"].lower(): d["score"] for d in raw}
        p_pos = probs.get("positive", 0.0)
        p_neg = probs.get("negative", 0.0)
        p_neu = probs.get("neutral", 0.0)
        return SentimentResult(polarity=p_pos - p_neg, p_negative=p_neg, p_neutral=p_neu, p_positive=p_pos)

    def score_batch(self, texts: list[str]) -> list[SentimentResult]:
        return [self.score(t) for t in texts]


def intensity_weight(metadata: dict | None) -> float:
    """iota_t in PDF Eq.1: engagement/reach-based intensity for social-media inputs.
    Real engagement metrics (likes/views/shares) are not currently pulled by the DAL for
    any configured source, so this returns the documented default of 1.0 (neutral weight)
    rather than fabricating an engagement estimate. Extend once a source exposes real
    engagement metadata (e.g. YouTube `statistics.viewCount` if wired into the DAL)."""
    metadata = metadata or {}
    view_count = metadata.get("view_count")
    if view_count is None:
        return 1.0
    # simple, documented, monotonic squashing of raw view count into (0, 1]; not claimed
    # to be a validated engagement model, just an honest placeholder once real data exists.
    import math

    return min(1.0, math.log10(max(view_count, 1) + 1) / 8.0)
