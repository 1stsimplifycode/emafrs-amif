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


def one_way_anova_f(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way ANOVA F-statistic: ratio of between-group to within-group variance
    (both mean-square-normalized by degrees of freedom). Implemented directly (not just
    calling scipy) so it can be recomputed cheaply inside a permutation loop without
    scipy's per-call overhead of re-validating/re-splitting groups each time."""
    unique_groups = np.unique(groups)
    grand_mean = values.mean()
    ss_between = sum(
        (groups == g).sum() * (values[groups == g].mean() - grand_mean) ** 2 for g in unique_groups
    )
    ss_within = sum(((values[groups == g] - values[groups == g].mean()) ** 2).sum() for g in unique_groups)
    df_between = len(unique_groups) - 1
    df_within = len(values) - len(unique_groups)
    if ss_within == 0 or df_within <= 0:
        return float("inf") if ss_between > 0 else 0.0
    return (ss_between / df_between) / (ss_within / df_within)


@dataclass
class PermutationAnovaResult:
    observed_f: float
    p_value: float
    n_permutations: int
    between_group_variance: float
    mean_within_group_variance: float


def permutation_test_anova(
    values: np.ndarray, groups: np.ndarray, n_permutations: int = 5000, seed: int = 0
) -> PermutationAnovaResult:
    """Tests whether GROUPS explain more variance in VALUES than chance grouping would,
    without assuming normality (unlike scipy's parametric f_oneway p-value, which this
    does not replace but can be cross-referenced against) -- appropriate when group sizes
    are small/unequal, as with per-regime sample counts in this project's real data.
    p_value = fraction of label-shuffled F-statistics >= the observed F-statistic."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    observed_f = one_way_anova_f(values, groups)

    rng = np.random.default_rng(seed)
    null_f = np.empty(n_permutations)
    for i in range(n_permutations):
        null_f[i] = one_way_anova_f(values, rng.permutation(groups))
    p_value = float((null_f >= observed_f).mean())

    unique_groups = np.unique(groups)
    group_means = [values[groups == g].mean() for g in unique_groups]
    group_vars = [values[groups == g].var() for g in unique_groups]

    return PermutationAnovaResult(
        observed_f=float(observed_f),
        p_value=p_value,
        n_permutations=n_permutations,
        between_group_variance=float(np.var(group_means)),
        mean_within_group_variance=float(np.mean(group_vars)),
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
