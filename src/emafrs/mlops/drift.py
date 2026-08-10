"""Data drift detection and retraining recommendation -- build brief Section 35.

Uses a two-sample Kolmogorov-Smirnov test per feature, comparing a model's original
training-window distribution against a recent live window -- a standard, well-understood
drift-detection technique, not a novel claim. Retraining is only ever RECOMMENDED here,
never triggered automatically (build brief Section 35: "do not automatically replace
production models without safeguards").
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy import stats


@dataclass
class DriftReport:
    feature_drift_pvalues: dict[str, float]
    drifted_features: list[str]
    drift_fraction: float
    retrain_recommended: bool


def detect_feature_drift(
    reference: pd.DataFrame,
    recent: pd.DataFrame,
    alpha: float = 0.01,
    drift_fraction_threshold: float = 0.2,
) -> DriftReport:
    """`reference`: the training-window feature distribution. `recent`: a recent window of
    the same features. A feature is flagged "drifted" if its two-sample KS test p-value
    < alpha (deliberately strict to reduce false positives from routine market noise --
    with many features tested, some will cross a looser threshold by chance alone).
    Retraining is recommended only if more than `drift_fraction_threshold` of features
    have drifted; a single drifted feature is common and not by itself actionable.
    """
    pvalues: dict[str, float] = {}
    for col in reference.columns:
        if col not in recent.columns:
            continue
        ref_vals = reference[col].dropna().to_numpy()
        recent_vals = recent[col].dropna().to_numpy()
        if len(ref_vals) < 10 or len(recent_vals) < 10:
            continue
        _, p = stats.ks_2samp(ref_vals, recent_vals)
        pvalues[col] = float(p)

    drifted = [c for c, p in pvalues.items() if p < alpha]
    fraction = len(drifted) / len(pvalues) if pvalues else 0.0

    return DriftReport(
        feature_drift_pvalues=pvalues,
        drifted_features=drifted,
        drift_fraction=fraction,
        retrain_recommended=fraction > drift_fraction_threshold,
    )
