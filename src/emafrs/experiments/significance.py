"""Paired statistical significance testing across walk-forward folds/seeds -- build brief
Section 24: "do not optimize for one impressive metric... statistical significance tests."
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class SignificanceResult:
    metric_name: str
    mean_a: float
    mean_b: float
    mean_diff: float
    std_diff: float
    n_pairs: int
    wilcoxon_stat: float | None
    wilcoxon_p: float | None
    paired_t_stat: float | None
    paired_t_p: float | None
    significant_at_05: bool


def compare_paired_results(
    values_a: list[float], values_b: list[float], metric_name: str, alpha: float = 0.05
) -> SignificanceResult:
    """Paired comparison (same folds/seeds, in the same order) of two models' metric
    values. Reports both the non-parametric Wilcoxon signed-rank test (robust to the small
    sample sizes typical with a handful of walk-forward folds) and the paired t-test.
    Callers are responsible for ensuring values_a/values_b are genuinely paired (identical
    fold/seed ordering) -- this function has no way to verify that itself."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"Paired comparison requires equal-length arrays, got {len(a)} vs {len(b)}")
    diff = a - b

    wilcoxon_stat = wilcoxon_p = None
    if len(a) >= 6 and not np.allclose(diff, 0):
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(a, b)
        except ValueError:
            pass

    t_stat = t_p = None
    if len(a) >= 2 and not np.allclose(diff, diff[0]):
        t_stat, t_p = stats.ttest_rel(a, b)

    sig = bool((wilcoxon_p is not None and wilcoxon_p < alpha) or (t_p is not None and t_p < alpha))

    return SignificanceResult(
        metric_name=metric_name,
        mean_a=float(a.mean()), mean_b=float(b.mean()),
        mean_diff=float(diff.mean()), std_diff=float(diff.std()), n_pairs=len(a),
        wilcoxon_stat=float(wilcoxon_stat) if wilcoxon_stat is not None else None,
        wilcoxon_p=float(wilcoxon_p) if wilcoxon_p is not None else None,
        paired_t_stat=float(t_stat) if t_stat is not None else None,
        paired_t_p=float(t_p) if t_p is not None else None,
        significant_at_05=sig,
    )


def confidence_interval(values: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """Returns (mean, lower, upper) using a t-distribution CI -- appropriate for the small
    sample sizes (a handful of folds/seeds) typical in this project's experiments, rather
    than assuming normality justifies a z-interval."""
    a = np.asarray(values, dtype=float)
    n = len(a)
    if n < 2:
        return float(a.mean()) if n else float("nan"), float("nan"), float("nan")
    mean = a.mean()
    sem = stats.sem(a)
    margin = sem * stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return float(mean), float(mean - margin), float(mean + margin)
