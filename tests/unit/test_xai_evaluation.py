import numpy as np

from emafrs.xai.evaluation import (
    counterfactual_validity_rate,
    faithfulness_score,
    sparsity_at_coverage,
    stability_score,
)


def test_faithfulness_score_positive_when_attribution_correctly_identifies_influential_features():
    """A linear predict_proba where only features 0 and 1 matter; attribution correctly
    points at them -- removing them should hurt more than removing random features."""
    weights = np.array([5.0, 5.0, 0.0, 0.0, 0.0])

    def predict_proba_fn(x: np.ndarray) -> np.ndarray:
        score = float(np.dot(weights, x))
        p1 = 1 / (1 + np.exp(-score))
        return np.array([1 - p1, p1])

    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    correct_attribution = np.array([5.0, 5.0, 0.0, 0.0, 0.0])  # matches the true weights
    score = faithfulness_score(predict_proba_fn, x, correct_attribution, target_class=1, top_k=2, n_random_trials=30, seed=0)
    assert score > 0  # top-attributed features degrade the prediction more than random removal


def test_faithfulness_score_near_zero_for_uninformative_attribution():
    weights = np.array([5.0, 5.0, 0.0, 0.0, 0.0])

    def predict_proba_fn(x: np.ndarray) -> np.ndarray:
        score = float(np.dot(weights, x))
        p1 = 1 / (1 + np.exp(-score))
        return np.array([1 - p1, p1])

    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    wrong_attribution = np.array([0.0, 0.0, 0.0, 5.0, 5.0])  # points at features that DON'T matter
    score = faithfulness_score(predict_proba_fn, x, wrong_attribution, target_class=1, top_k=2, n_random_trials=30, seed=0)
    assert score < 0.05  # should not meaningfully outperform random removal


def test_stability_score_high_for_smooth_linear_function():
    weights = np.array([1.0, -1.0, 0.5])

    def attribution_fn(x: np.ndarray) -> np.ndarray:
        return weights * x  # smooth, deterministic "attribution"

    x = np.array([2.0, 3.0, -1.0])
    score = stability_score(attribution_fn, x, n_perturbations=15, noise_std=0.01, seed=0)
    assert score > 0.8


def test_stability_score_low_for_noisy_random_attribution():
    rng_global = np.random.default_rng(42)

    def attribution_fn(x: np.ndarray) -> np.ndarray:
        return rng_global.normal(0, 1, size=x.shape)  # ignores x entirely -- pure noise

    x = np.array([2.0, 3.0, -1.0, 0.5, 1.2])
    score = stability_score(attribution_fn, x, n_perturbations=15, noise_std=0.01, seed=0)
    assert score < 0.5


def test_sparsity_at_coverage_counts_correctly():
    # one dominant feature carrying 90% of total attribution magnitude
    attributions = np.array([9.0, 0.3, 0.3, 0.2, 0.2])
    n_needed = sparsity_at_coverage(attributions, coverage=0.8)
    assert n_needed == 1  # the single dominant feature alone covers >80%


def test_sparsity_at_coverage_needs_more_features_when_distributed_evenly():
    attributions = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    n_needed = sparsity_at_coverage(attributions, coverage=0.8)
    assert n_needed >= 4  # evenly spread -- needs most features to reach 80%


def test_counterfactual_validity_rate_over_multiple_samples():
    def predict_fn(x: np.ndarray) -> int:
        return int(x[0] + x[1] > 0)

    X = np.array([[-1.0, 0.0], [-100.0, 0.0], [0.5, 0.5]])  # last one already predicts 1
    report = counterfactual_validity_rate(predict_fn, X, feature_names=["f0", "f1"], search_range=(-3.0, 3.0), n_steps=61)
    assert report.n_samples == 3
    assert 0.0 <= report.validity_rate <= 1.0
    assert report.n_flipped >= 1  # at least the first sample (small delta needed) should flip
