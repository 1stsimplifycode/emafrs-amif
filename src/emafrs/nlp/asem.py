"""Affective Signal Extraction Module (ASEM) -- PDF Section 6.4.

Assembles the structured affective feature vector a_t = [e_t || s_t || i_t || c_t || rho_t]
(Eq. 1) from a single communication: 7-category emotion vector, sentiment polarity,
intensity weight, semantic embedding, and role one-hot -- plus the derived credibility
score and emotional hype score (Sec 6.4, Fig. 2).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from emafrs.nlp.credibility import CredibilityScore, compute_credibility
from emafrs.nlp.emotion import EmotionScores, ZeroShotEmotionDetector
from emafrs.nlp.hype import compute_hype_score
from emafrs.nlp.roles import infer_role, role_to_onehot
from emafrs.nlp.semantic_encoder import ContextualSemanticEncoder
from emafrs.nlp.sentiment import SentimentPolarityClassifier, SentimentResult, intensity_weight


@dataclass
class AffectiveFeatureVector:
    text: str
    provider: str
    entity: str | None
    timestamp: datetime | None
    emotion: EmotionScores
    sentiment: SentimentResult
    intensity: float
    semantic_embedding: np.ndarray
    role: str
    role_onehot: list[int]
    credibility: CredibilityScore
    hype_score: float

    def as_vector(self) -> np.ndarray:
        """Concatenation per PDF Eq. (1): a_t = [e_t || s_t || i_t || c_t || rho_t]."""
        return np.concatenate(
            [
                np.array(self.emotion.as_vector(), dtype=float),
                np.array([self.sentiment.polarity], dtype=float),
                np.array([self.intensity], dtype=float),
                self.semantic_embedding.astype(float),
                np.array(self.role_onehot, dtype=float),
            ]
        )


class AffectiveSignalExtractionModule:
    """Loads all sub-models once (nontrivial cost on CPU) and exposes `.process()` per
    communication. Model choices come from configs/nlp_models.yaml / the benchmark results
    in docs/NLP_MODEL_BENCHMARK.md, not picked arbitrarily."""

    def __init__(
        self,
        sentiment_model: str = "ProsusAI/finbert",
        emotion_model: str = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
        semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self._sentiment = SentimentPolarityClassifier(sentiment_model)
        self._emotion = ZeroShotEmotionDetector(emotion_model)
        self._semantic = ContextualSemanticEncoder(semantic_model)

    def process(
        self,
        text: str,
        provider: str,
        entity: str | None = None,
        timestamp: datetime | None = None,
        metadata: dict | None = None,
        historical_impact_corr: float | None = None,
    ) -> AffectiveFeatureVector:
        metadata = metadata or {}
        emotion_scores = self._emotion.score(text)
        sentiment_result = self._sentiment.score(text)
        intensity = intensity_weight(metadata)
        embedding = self._semantic.encode_one(text)
        role = infer_role(provider, metadata)
        role_onehot = role_to_onehot(role)
        credibility = compute_credibility(role, historical_impact_corr)
        hype = compute_hype_score(credibility.value, emotion_scores)

        return AffectiveFeatureVector(
            text=text,
            provider=provider,
            entity=entity,
            timestamp=timestamp,
            emotion=emotion_scores,
            sentiment=sentiment_result,
            intensity=intensity,
            semantic_embedding=embedding,
            role=role,
            role_onehot=role_onehot,
            credibility=credibility,
            hype_score=hype,
        )

    def process_batch(self, records: list[dict]) -> list[AffectiveFeatureVector]:
        return [self.process(**r) for r in records]
