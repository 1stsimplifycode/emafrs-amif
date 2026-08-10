import pytest

from emafrs.experiments.significance import compare_paired_results, confidence_interval


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
