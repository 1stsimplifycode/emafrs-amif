import numpy as np

from emafrs.xai.evaluation import (
    comprehensiveness,
    counterfactual_validity_rate,
    deletion_auc,
    faithfulness_score,
    insertion_auc,
    sparsity_at_coverage,
    stability_score,
    sufficiency,
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


def _linear_predict_proba(weights: np.ndarray):
    def predict_proba_fn(x: np.ndarray) -> np.ndarray:
        score = float(np.dot(weights, x))
        p1 = 1 / (1 + np.exp(-score))
        return np.array([1 - p1, p1])
    return predict_proba_fn


def test_deletion_auc_lower_for_correct_than_uninformative_attribution():
    """Removing the truly-influential features first should crash probability faster
    (lower AUC) than removing uninformative ones first."""
    weights = np.array([5.0, 5.0, 0.0, 0.0, 0.0])
    predict_proba_fn = _linear_predict_proba(weights)
    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    correct_attr = np.array([5.0, 5.0, 0.01, 0.01, 0.01])
    wrong_attr = np.array([0.01, 0.01, 0.01, 5.0, 5.0])

    auc_correct = deletion_auc(predict_proba_fn, x, correct_attr, target_class=1)
    auc_wrong = deletion_auc(predict_proba_fn, x, wrong_attr, target_class=1)
    assert auc_correct < auc_wrong


def test_insertion_auc_higher_for_correct_than_uninformative_attribution():
    """Revealing the truly-influential features first should raise probability faster
    (higher AUC) than revealing uninformative ones first."""
    weights = np.array([5.0, 5.0, 0.0, 0.0, 0.0])
    predict_proba_fn = _linear_predict_proba(weights)
    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    correct_attr = np.array([5.0, 5.0, 0.01, 0.01, 0.01])
    wrong_attr = np.array([0.01, 0.01, 0.01, 5.0, 5.0])

    auc_correct = insertion_auc(predict_proba_fn, x, correct_attr, target_class=1)
    auc_wrong = insertion_auc(predict_proba_fn, x, wrong_attr, target_class=1)
    assert auc_correct > auc_wrong


def test_comprehensiveness_positive_when_top_features_are_truly_influential():
    weights = np.array([5.0, 5.0, 0.0, 0.0, 0.0])
    predict_proba_fn = _linear_predict_proba(weights)
    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    correct_attr = np.array([5.0, 5.0, 0.01, 0.01, 0.01])
    score = comprehensiveness(predict_proba_fn, x, correct_attr, target_class=1, top_k=2)
    assert score > 0.1  # removing the two truly-influential features should meaningfully drop confidence


def test_sufficiency_small_gap_when_top_features_alone_reproduce_full_prediction():
    weights = np.array([5.0, 5.0, 0.0, 0.0, 0.0])
    predict_proba_fn = _linear_predict_proba(weights)
    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    correct_attr = np.array([5.0, 5.0, 0.01, 0.01, 0.01])
    # keeping only features 0,1 (the ONLY ones with nonzero weight) must reproduce the
    # full prediction exactly, since features 2-4 contribute nothing regardless
    score = sufficiency(predict_proba_fn, x, correct_attr, target_class=1, top_k=2)
    assert abs(score) < 1e-6


def test_counterfactual_validity_rate_over_multiple_samples():
    def predict_fn(x: np.ndarray) -> int:
        return int(x[0] + x[1] > 0)

    X = np.array([[-1.0, 0.0], [-100.0, 0.0], [0.5, 0.5]])  # last one already predicts 1
    report = counterfactual_validity_rate(predict_fn, X, feature_names=["f0", "f1"], search_range=(-3.0, 3.0), n_steps=61)
    assert report.n_samples == 3
    assert 0.0 <= report.validity_rate <= 1.0
    assert report.n_flipped >= 1  # at least the first sample (small delta needed) should flip
