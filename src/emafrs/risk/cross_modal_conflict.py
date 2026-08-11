"""Cross-modal agreement/conflict detection -- build brief Section 20.

Example from the build brief: Financial -> Bullish, News -> Bearish, Emotion -> Fear,
Microstructure -> Strong Buying. This module quantifies whether per-modality directional
signals agree or conflict; it does NOT claim to explain WHY they conflict, and conflict is
reported as "modality disagreement," never as evidence of manipulation or causal insight
on its own (see docs/LIMITATIONS.md and the project's anti-overclaiming rule on
correlation-vs-causation for affective/microstructure signals).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConflictReport:
    modality_directions: dict[str, float]  # each in [-1, +1]; sign = direction, magnitude = strength
    pairwise_agreement: dict[tuple[str, str], float]  # +1 fully agree, -1 fully oppose, 0 ambiguous
    mean_agreement: float
    most_conflicted_pair: tuple[str, str] | None
    is_conflicted: bool


def modality_direction_from_attributions(attributions: dict) -> dict[str, float]:
    """Derives a REAL per-modality directional signal from Integrated Gradients
    attributions already computed for a live prediction (see
    emafrs.xai.integrated_gradients.compute_integrated_gradients) -- the sum of that
    modality's attribution values toward the model's predicted class, squashed to
    [-1, 1] via tanh (attribution sums are unbounded; tanh gives a bounded, comparable
    "direction strength" without inventing a probability). Positive = this modality's
    evidence net pushed toward the predicted class; negative = net pushed away.

    This is model-attributed evidence, not a claim about true market direction (same
    anti-overclaiming rule as every other attribution in this project) -- but unlike a
    fabricated placeholder, it is computed from real numbers already produced by the
    live inference pipeline for this specific prediction, not asserted.

    `attributions`: a dict of modality name -> attribution array, exactly the dict
    `compute_integrated_gradients` returns (after removing the "regime" entry, which is
    not a modality)."""
    return {name: float(np.tanh(np.sum(arr))) for name, arr in attributions.items()}


def compute_agreement(
    modality_directions: dict[str, float], conflict_threshold: float = 0.0, neutral_band: float = 0.05
) -> ConflictReport:
    """`neutral_band`: directions with |value| below this are treated as ambiguous/neutral
    (contribute 0 to pairwise agreement rather than an arbitrary sign) rather than forcing
    a near-zero value to count as a strong directional vote.

    `is_conflicted` is True when mean pairwise agreement is at or below `conflict_threshold`
    (default 0.0: net disagreement across modality pairs)."""
    names = list(modality_directions.keys())
    n = len(names)
    pairwise: dict[tuple[str, str], float] = {}

    for i in range(n):
        for j in range(i + 1, n):
            a, b = modality_directions[names[i]], modality_directions[names[j]]
            if abs(a) < neutral_band or abs(b) < neutral_band:
                pairwise[(names[i], names[j])] = 0.0
            else:
                pairwise[(names[i], names[j])] = float(np.sign(a) * np.sign(b))

    mean_agreement = float(np.mean(list(pairwise.values()))) if pairwise else 0.0
    most_conflicted = min(pairwise, key=pairwise.get) if pairwise else None
    if most_conflicted is not None and pairwise[most_conflicted] >= 0:
        most_conflicted = None  # no pair actually disagrees

    return ConflictReport(
        modality_directions=dict(modality_directions),
        pairwise_agreement=pairwise,
        mean_agreement=mean_agreement,
        most_conflicted_pair=most_conflicted,
        is_conflicted=mean_agreement <= conflict_threshold,
    )
