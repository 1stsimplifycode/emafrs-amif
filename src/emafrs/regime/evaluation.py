"""Regime quality evaluation -- the build brief explicitly warns "do not assume the HMM
automatically produces meaningful regimes." This module checks that assumption rather than
taking it on faith.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


@dataclass
class RegimeQualityReport:
    n_states: int
    state_counts: dict[int, int]
    mean_run_length: float
    monotonic_vol_ordering: bool
    vol_by_regime: dict[int, float]
    silhouette: float | None
    transition_matrix: list[list[float]]
    warnings: list[str] = field(default_factory=list)


def _mean_run_length(labels: pd.Series) -> float:
    """Average number of consecutive periods a regime persists before switching. A
    healthy regime detector should have runs meaningfully longer than 1 (else it is just
    noise-chasing every day's volatility wiggle, not detecting a structural regime)."""
    if len(labels) == 0:
        return 0.0
    changes = labels.ne(labels.shift()).cumsum()
    run_lengths = labels.groupby(changes).size()
    return float(run_lengths.mean())


def evaluate_regime_quality(
    regime_labels: pd.Series,
    features: pd.DataFrame,
    vol_column: str,
    transition_matrix: np.ndarray,
) -> RegimeQualityReport:
    warnings: list[str] = []
    aligned = features.loc[regime_labels.index]

    state_counts = regime_labels.value_counts().sort_index().to_dict()
    n_states = len(state_counts)

    vol_by_regime = aligned.groupby(regime_labels)[vol_column].mean().to_dict()
    ordered_regimes = sorted(vol_by_regime)
    monotonic = all(
        vol_by_regime[ordered_regimes[i]] <= vol_by_regime[ordered_regimes[i + 1]]
        for i in range(len(ordered_regimes) - 1)
    )
    if not monotonic:
        warnings.append(
            "Regime volatility ordering is not monotonic -- relabeling by mean vol at fit "
            "time should guarantee this on training data; a violation here likely means "
            "this evaluation was run on out-of-sample predictions where regime semantics drifted."
        )

    mean_run = _mean_run_length(regime_labels)
    if mean_run < 3:
        warnings.append(
            f"Mean regime run length is {mean_run:.1f} periods -- regimes are switching almost "
            "every observation, which suggests the HMM is fitting noise rather than a slow-moving "
            "structural state. Consider fewer states, more/less-noisy features, or a longer window."
        )

    silhouette = None
    if n_states > 1 and len(aligned) > n_states:
        try:
            feature_cols = [c for c in aligned.columns if aligned[c].notna().all()]
            if feature_cols:
                silhouette = float(silhouette_score(aligned[feature_cols].values, regime_labels.values))
                if silhouette < 0.1:
                    warnings.append(
                        f"Silhouette score {silhouette:.3f} is low -- regimes are not well-separated "
                        "in feature space; treat regime labels as a soft/noisy signal, not ground truth."
                    )
        except ValueError as exc:
            warnings.append(f"Silhouette score could not be computed: {exc}")

    smallest_regime_frac = min(state_counts.values()) / sum(state_counts.values())
    if smallest_regime_frac < 0.02:
        warnings.append(
            f"Smallest regime accounts for only {smallest_regime_frac:.1%} of observations -- "
            "treat its statistics as low-sample-size and unreliable in downstream evaluation."
        )

    return RegimeQualityReport(
        n_states=n_states,
        state_counts=state_counts,
        mean_run_length=mean_run,
        monotonic_vol_ordering=monotonic,
        vol_by_regime=vol_by_regime,
        silhouette=silhouette,
        transition_matrix=transition_matrix.tolist(),
        warnings=warnings,
    )
