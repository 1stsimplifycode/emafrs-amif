"""Counterfactual explainability -- PDF Sec 6.8, Claim 14, Claim 24:

  "computing the decision difference Decision(Mt,St,Rt) - Decision(Mt,St+delta,Rt) across a
  plurality of hypothetical perturbation values delta, and translating the resulting
  sensitivity profile into a ranked natural language explanation identifying the minimum
  signal change that would have prevented alert generation."

Implemented as an empirical grid search over the REAL trained model's decision boundary for
each feature -- the minimal-delta claim is searched for and verified against actual model
output, never asserted without running the model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class CounterfactualResult:
    feature_name: str
    original_value: float
    counterfactual_value: float | None
    delta: float | None
    original_prediction: int
    counterfactual_prediction: int | None
    flipped: bool


def search_minimal_counterfactual(
    predict_fn: Callable[[np.ndarray], int],
    x: np.ndarray,
    feature_idx: int,
    feature_name: str,
    search_range: tuple[float, float] = (-5.0, 5.0),
    n_steps: int = 41,
) -> CounterfactualResult:
    """Grid-searches perturbations of ONE feature (holding all others fixed) for the
    minimum-magnitude change that flips predict_fn's output relative to x's original
    prediction. Candidates are tried in order of increasing |delta| so the first flip found
    is the minimal one found by this search (not a global guarantee -- a finer grid could
    find a smaller delta -- but it is an honest, disclosed empirical search, not an
    analytic minimum)."""
    original_pred = predict_fn(x)
    candidates = np.linspace(search_range[0], search_range[1], n_steps)
    candidates = candidates[np.argsort(np.abs(candidates))]

    for delta in candidates:
        if delta == 0:
            continue
        x_perturbed = x.copy()
        x_perturbed[feature_idx] = x[feature_idx] + delta
        pred = predict_fn(x_perturbed)
        if pred != original_pred:
            return CounterfactualResult(
                feature_name=feature_name,
                original_value=float(x[feature_idx]),
                counterfactual_value=float(x_perturbed[feature_idx]),
                delta=float(delta),
                original_prediction=int(original_pred),
                counterfactual_prediction=int(pred),
                flipped=True,
            )

    return CounterfactualResult(
        feature_name=feature_name, original_value=float(x[feature_idx]), counterfactual_value=None,
        delta=None, original_prediction=int(original_pred), counterfactual_prediction=None, flipped=False,
    )


def search_all_feature_counterfactuals(
    predict_fn: Callable[[np.ndarray], int],
    x: np.ndarray,
    feature_names: list[str],
    search_range: tuple[float, float] = (-5.0, 5.0),
    n_steps: int = 41,
) -> list[CounterfactualResult]:
    """Returns counterfactuals for every feature that CAN flip the prediction within
    `search_range`, ranked by ascending |delta| -- the smallest, most "minimal" change
    first, matching Claim 24's requirement to report the minimum sufficient signal change."""
    results = [
        search_minimal_counterfactual(predict_fn, x, i, name, search_range, n_steps)
        for i, name in enumerate(feature_names)
    ]
    flipped = [r for r in results if r.flipped]
    return sorted(flipped, key=lambda r: abs(r.delta))
