import numpy as np
import pytest

from emafrs.experiments.significance import (
    compare_paired_results,
    confidence_interval,
    one_way_anova_f,
    permutation_test_anova,
)


def test_compare_paired_results_detects_clear_difference():
    values_a = [0.80, 0.82, 0.79, 0.81, 0.83, 0.80, 0.82, 0.81]
    values_b = [0.60, 0.61, 0.59, 0.62, 0.60, 0.61, 0.59, 0.60]
    result = compare_paired_results(values_a, values_b, metric_name="accuracy")
    assert result.mean_diff > 0
    assert result.significant_at_05 is True
    assert result.wilcoxon_p is not None and result.wilcoxon_p < 0.05


def test_compare_paired_results_no_difference_not_significant():
    values_a = [0.70, 0.71, 0.69, 0.70, 0.72, 0.68, 0.71, 0.70]
    values_b = [0.70, 0.69, 0.71, 0.70, 0.71, 0.69, 0.70, 0.70]
    result = compare_paired_results(values_a, values_b, metric_name="accuracy")
    assert abs(result.mean_diff) < 0.02
    assert result.significant_at_05 is False


def test_compare_paired_results_requires_equal_length():
    with pytest.raises(ValueError):
        compare_paired_results([0.1, 0.2], [0.1, 0.2, 0.3], metric_name="x")


def test_confidence_interval_contains_mean():
    values = [0.70, 0.72, 0.71, 0.69, 0.73, 0.70]
    mean, lo, hi = confidence_interval(values)
    assert lo <= mean <= hi
    assert lo < hi


def test_confidence_interval_degenerate_single_value():
    mean, lo, hi = confidence_interval([0.5])
    assert mean == 0.5
    assert lo != lo  # NaN


def test_one_way_anova_f_zero_for_identical_group_means_and_spread():
    values = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    groups = np.array([1, 1, 1, 2, 2, 2])
    assert one_way_anova_f(values, groups) == 0.0


def test_one_way_anova_f_large_for_clearly_separated_groups():
    rng = np.random.default_rng(0)
    group_a = rng.normal(0.0, 0.01, size=30)
    group_b = rng.normal(5.0, 0.01, size=30)
    values = np.concatenate([group_a, group_b])
    groups = np.array([1] * 30 + [2] * 30)
    f = one_way_anova_f(values, groups)
    assert f > 1000  # tiny within-group spread, huge between-group gap


def test_permutation_test_anova_significant_for_clearly_separated_groups():
    rng = np.random.default_rng(1)
    group_a = rng.normal(0.0, 0.01, size=30)
    group_b = rng.normal(5.0, 0.01, size=30)
    values = np.concatenate([group_a, group_b])
    groups = np.array([1] * 30 + [2] * 30)
    result = permutation_test_anova(values, groups, n_permutations=500, seed=0)
    assert result.p_value < 0.01
    assert result.between_group_variance > result.mean_within_group_variance


def test_permutation_test_anova_not_significant_for_random_labels_on_homogeneous_data():
    rng = np.random.default_rng(2)
    values = rng.normal(0.0, 1.0, size=60)  # no real group structure at all
    groups = rng.integers(0, 3, size=60)
    result = permutation_test_anova(values, groups, n_permutations=500, seed=0)
    assert result.p_value > 0.05
