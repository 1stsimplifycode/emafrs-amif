"""Quantitative XAI evaluation -- build brief Section 21/XAI_EVALUATION.md: faithfulness,
stability, sparsity, counterfactual validity. XAI is evaluated as a measurable property,
not presented only as visualization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def _trapezoid_auc(y: list[float], dx: float) -> float:
    """Trapezoidal AUC without depending on numpy.trapz/trapezoid (renamed/removed across
    numpy versions -- avoids a version-dependent import)."""
    y = np.asarray(y, dtype=float)
    return float(np.sum((y[:-1] + y[1:]) / 2.0 * dx))


def faithfulness_score(
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    attributions: np.ndarray,
    target_class: int,
    top_k: int = 5,
    n_random_trials: int = 20,
    seed: int = 0,
) -> float:
    """Removes (zeros) the top_k most-attributed features and measures the drop in
    predicted probability for target_class, MINUS the mean drop from removing top_k
    RANDOM features (same count). A faithful attribution should degrade the prediction
    more than an equal-size random removal -- there is no principled absolute threshold,
    so this is always reported relative to the random baseline."""
    baseline_proba = predict_proba_fn(x)[target_class]

    top_idx = np.argsort(-np.abs(attributions))[:top_k]
    x_top_removed = x.copy()
    x_top_removed[top_idx] = 0.0
    top_drop = baseline_proba - predict_proba_fn(x_top_removed)[target_class]

    rng = np.random.default_rng(seed)
    random_drops = []
    for _ in range(n_random_trials):
        rand_idx = rng.choice(len(x), size=min(top_k, len(x)), replace=False)
        x_rand_removed = x.copy()
        x_rand_removed[rand_idx] = 0.0
        random_drops.append(baseline_proba - predict_proba_fn(x_rand_removed)[target_class])

    return float(top_drop - np.mean(random_drops))


def deletion_auc(
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    attributions: np.ndarray,
    target_class: int,
) -> float:
    """Deletion metric (Petsiuk et al. 2018 RISE): progressively zero out features in
    DESCENDING |attribution| order (most-attributed first) and trace the predicted
    probability for target_class after each removal. Returns the area under that curve
    (trapezoidal, x-axis = fraction of features removed 0..1). A FAITHFUL attribution
    should crash the probability early (most-important features removed first), giving a
    LOW AUC. A random/uninformative attribution gives an AUC close to the curve of
    removing features in random order. This is a full-curve generalization of
    `faithfulness_score`'s single top-k point."""
    order = np.argsort(-np.abs(attributions))
    n = len(x)
    x_work = x.copy()
    probs = [predict_proba_fn(x_work)[target_class]]
    for idx in order:
        x_work = x_work.copy()
        x_work[idx] = 0.0
        probs.append(predict_proba_fn(x_work)[target_class])
    return _trapezoid_auc(probs, dx=1.0 / n)


def insertion_auc(
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    attributions: np.ndarray,
    target_class: int,
) -> float:
    """Insertion metric (Petsiuk et al. 2018): the mirror of deletion -- start from an
    all-zero (baseline) input and progressively REVEAL features in descending
    |attribution| order, tracing predicted probability as evidence is added back. Returns
    the AUC. A faithful attribution should raise the probability quickly (most-important
    features revealed first), giving a HIGH AUC."""
    order = np.argsort(-np.abs(attributions))
    n = len(x)
    x_work = np.zeros_like(x)
    probs = [predict_proba_fn(x_work)[target_class]]
    for idx in order:
        x_work = x_work.copy()
        x_work[idx] = x[idx]
        probs.append(predict_proba_fn(x_work)[target_class])
    return _trapezoid_auc(probs, dx=1.0 / n)


def comprehensiveness(
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    attributions: np.ndarray,
    target_class: int,
    top_k: int = 5,
) -> float:
    """ERASER-style comprehensiveness: how much predicted probability DROPS when the
    top_k attributed features are removed, with NO random-baseline subtraction (unlike
    `faithfulness_score`) -- the raw magnitude of the claimed evidence's importance.
    Higher = the attributed features were more necessary to the prediction."""
    baseline_proba = predict_proba_fn(x)[target_class]
    top_idx = np.argsort(-np.abs(attributions))[:top_k]
    x_removed = x.copy()
    x_removed[top_idx] = 0.0
    return float(baseline_proba - predict_proba_fn(x_removed)[target_class])


def sufficiency(
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    attributions: np.ndarray,
    target_class: int,
    top_k: int = 5,
) -> float:
    """ERASER-style sufficiency: how CLOSE the prediction using ONLY the top_k attributed
    features (everything else zeroed) is to the prediction using the full input --
    returned as (full_proba - top_k_only_proba), so LOWER is more sufficient (the top_k
    features alone nearly reproduce the full prediction)."""
    baseline_proba = predict_proba_fn(x)[target_class]
    top_idx = np.argsort(-np.abs(attributions))[:top_k]
    x_only_top = np.zeros_like(x)
    x_only_top[top_idx] = x[top_idx]
    return float(baseline_proba - predict_proba_fn(x_only_top)[target_class])


def stability_score(
    attribution_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    n_perturbations: int = 10,
    noise_std: float = 0.01,
    seed: int = 0,
) -> float:
    """Mean Spearman rank correlation between the attribution ranking on x and on small
    Gaussian perturbations of x. An explanation whose feature ranking flips under tiny
    input noise is not a stable/reliable explanation, regardless of how it looks on any
    single input."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    base_attr = attribution_fn(x)
    correlations = []
    for _ in range(n_perturbations):
        x_perturbed = x + rng.normal(0, noise_std, size=x.shape)
        perturbed_attr = attribution_fn(x_perturbed)
        corr, _ = spearmanr(base_attr, perturbed_attr)
        if not np.isnan(corr):
            correlations.append(corr)
    return float(np.mean(correlations)) if correlations else float("nan")


def sparsity_at_coverage(attributions: np.ndarray, coverage: float = 0.8) -> int:
    """Minimum number of (ranked by |attribution|) features whose cumulative absolute
    attribution reaches `coverage` fraction of the total -- fewer features needed = a
    sparser/more concise explanation for the same evidence coverage."""
    abs_attr = np.abs(attributions)
    total = abs_attr.sum()
    if total == 0:
        return len(attributions)
    order = np.argsort(-abs_attr)
    cumsum = np.cumsum(abs_attr[order])
    threshold = coverage * total
    return int(np.searchsorted(cumsum, threshold) + 1)


@dataclass
class CounterfactualValidityReport:
    n_samples: int
    n_flipped: int
    validity_rate: float
    mean_min_abs_delta: float | None


def counterfactual_validity_rate(
    predict_fn: Callable[[np.ndarray], int],
    X: np.ndarray,
    feature_names: list[str],
    search_range: tuple[float, float] = (-5.0, 5.0),
    n_steps: int = 41,
) -> CounterfactualValidityReport:
    """Fraction of samples for which SOME feature's counterfactual search finds a
    prediction-flipping perturbation within `search_range` -- if this is low, the model's
    decision surface is not well-explored by simple per-feature perturbations at this
    scale (informative on its own, not just a validity gate)."""
    from emafrs.xai.counterfactual import search_all_feature_counterfactuals

    n_flipped = 0
    min_deltas = []
    for i in range(len(X)):
        results = search_all_feature_counterfactuals(predict_fn, X[i], feature_names, search_range, n_steps)
        if results:
            n_flipped += 1
            min_deltas.append(abs(results[0].delta))

    rate = n_flipped / len(X) if len(X) else 0.0
    mean_delta = float(np.mean(min_deltas)) if min_deltas else None
    return CounterfactualValidityReport(
        n_samples=len(X), n_flipped=n_flipped, validity_rate=rate, mean_min_abs_delta=mean_delta
    )
