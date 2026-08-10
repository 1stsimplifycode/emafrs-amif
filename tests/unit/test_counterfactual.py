import numpy as np

from emafrs.xai.counterfactual import search_all_feature_counterfactuals, search_minimal_counterfactual


def _threshold_predict_fn(x: np.ndarray) -> int:
    """A simple, fully-known decision rule: predict 1 if x[0] + x[1] > 0 else 0. Lets us
    verify the search finds the mathematically correct minimal counterfactual."""
    return int(x[0] + x[1] > 0)


def test_search_finds_a_flip_when_one_exists():
    x = np.array([-2.0, 0.0])  # sum = -2, predicts 0
    result = search_minimal_counterfactual(_threshold_predict_fn, x, feature_idx=0, feature_name="f0",
                                            search_range=(-1.0, 5.0), n_steps=61)
    assert result.flipped is True
    assert result.counterfactual_prediction == 1
    # minimal delta to cross zero from x[0]=-2 is just above +2 (since x[1]=0)
    assert result.delta > 2.0
    assert result.delta < 2.2


def test_search_reports_no_flip_when_out_of_range():
    x = np.array([-100.0, 0.0])
    result = search_minimal_counterfactual(_threshold_predict_fn, x, feature_idx=0, feature_name="f0",
                                            search_range=(-1.0, 1.0), n_steps=21)
    assert result.flipped is False
    assert result.counterfactual_value is None
    assert result.delta is None


def test_search_all_features_ranks_by_minimal_delta():
    """`_threshold_predict_fn` (x[0]+x[1]>0) is symmetric in both features, so perturbing
    either by the same raw delta has an identical effect -- not useful for testing ranking.
    Uses an asymmetrically-weighted rule instead so the two features genuinely need
    different minimal deltas to flip."""

    def _weighted_predict_fn(x: np.ndarray) -> int:
        return int(x[0] + 3 * x[1] > 0)

    x = np.array([0.5, -1.0])  # weighted sum = 0.5 - 3 = -2.5
    # f0 needs delta > 2.5 to flip; f1 needs delta > 2.5/3 ~= 0.833 to flip -- f1 is smaller
    results = search_all_feature_counterfactuals(
        _weighted_predict_fn, x, feature_names=["f0", "f1"], search_range=(-4.0, 4.0), n_steps=161
    )
    assert len(results) == 2
    assert results[0].feature_name == "f1"  # smaller |delta| ranked first
    assert abs(results[0].delta) < abs(results[1].delta)


def test_search_all_features_excludes_non_flipping_features():
    def _ignore_second_feature(x: np.ndarray) -> int:
        return int(x[0] > 0)  # x[1] can never flip the decision

    x = np.array([-1.0, 0.0])
    results = search_all_feature_counterfactuals(
        _ignore_second_feature, x, feature_names=["f0", "f1"], search_range=(-2.0, 2.0), n_steps=41
    )
    assert len(results) == 1
    assert results[0].feature_name == "f0"
