"""Zero-shot emotion detection over the PDF's 7-category financial emotion taxonomy
(PDF Section 6.4.2): Confidence, Uncertainty, Optimism, Fear, Caution, Assertiveness,
Evasiveness.

HONESTY NOTE: no labeled dataset exists for this exact taxonomy, so this is NOT a
fine-tuned classifier -- it is zero-shot NLI-based classification (does this text entail
"This text expresses <emotion>"?). Every output surfaced downstream (dashboard, XAI,
model cards) must describe this as zero-shot, not as "trained emotion detection." See
configs/nlp_models.yaml `emotion_zero_shot` for the model and rationale.
"""
from __future__ import annotations

from dataclasses import dataclass

EMOTION_CATEGORIES = [
    "confidence", "uncertainty", "optimism", "fear", "caution", "assertiveness", "evasiveness",
]

_HYPOTHESIS_TEMPLATE = "This statement conveys {}."


@dataclass
class EmotionScores:
    scores: dict[str, float]  # category -> [0,1] entailment-derived score, independent (multi-label)
    method: str = "zero_shot_nli"

    def as_vector(self) -> list[float]:
        return [self.scores[c] for c in EMOTION_CATEGORIES]


class ZeroShotEmotionDetector:
    def __init__(self, hf_id: str = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"):
        from transformers import pipeline

        self.hf_id = hf_id
        self._pipe = pipeline("zero-shot-classification", model=hf_id)

    def score(self, text: str) -> EmotionScores:
        """Multi-label: each emotion category is scored independently (a communication can
        simultaneously convey both confidence and caution), matching PDF Sec 6.4.2's
        multi-label emotion vector e_t, not a single-label classification."""
        result = self._pipe(
            text,
            candidate_labels=EMOTION_CATEGORIES,
            hypothesis_template=_HYPOTHESIS_TEMPLATE,
            multi_label=True,
        )
        scores = dict(zip(result["labels"], result["scores"]))
        # restore canonical category order
        ordered = {c: scores[c] for c in EMOTION_CATEGORIES}
        return EmotionScores(scores=ordered)

    def score_batch(self, texts: list[str]) -> list[EmotionScores]:
        return [self.score(t) for t in texts]
