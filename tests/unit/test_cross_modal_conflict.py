import numpy as np

from emafrs.risk.cross_modal_conflict import compute_agreement, modality_direction_from_attributions


def test_modality_direction_positive_attribution_sum_gives_positive_direction():
    attrs = {"financial": np.array([[0.5, 0.3, -0.1]])}
    directions = modality_direction_from_attributions(attrs)
    assert directions["financial"] > 0


def test_modality_direction_negative_attribution_sum_gives_negative_direction():
    attrs = {"financial": np.array([[-0.5, -0.3, 0.1]])}
    directions = modality_direction_from_attributions(attrs)
    assert directions["financial"] < 0


def test_modality_direction_is_bounded_in_unit_interval_even_for_large_sums():
    attrs = {"financial": np.array([[1000.0, 1000.0]])}
    directions = modality_direction_from_attributions(attrs)
    assert -1.0 <= directions["financial"] <= 1.0  # tanh saturates to exactly +-1.0 at float precision limits, never beyond


def test_modality_direction_zero_sum_gives_zero_direction():
    attrs = {"financial": np.array([[0.5, -0.5]])}
    directions = modality_direction_from_attributions(attrs)
    assert directions["financial"] == 0.0


def test_modality_direction_covers_every_modality_key_given():
    attrs = {"financial": np.array([[0.5]]), "microstructure": np.array([[-0.5]])}
    directions = modality_direction_from_attributions(attrs)
    assert set(directions.keys()) == {"financial", "microstructure"}


def test_modality_direction_feeds_directly_into_compute_agreement():
    """End-to-end real-data path: attributions -> directions -> agreement report."""
    attrs = {"financial": np.array([[0.5, 0.3]]), "microstructure": np.array([[0.4, 0.2]])}
    directions = modality_direction_from_attributions(attrs)
    report = compute_agreement(directions)
    assert report.mean_agreement == 1.0  # both modalities pushed the same direction
    assert report.is_conflicted is False


def test_full_agreement_across_modalities():
    directions = {"financial": 0.8, "text": 0.6, "microstructure": 0.9}
    report = compute_agreement(directions)
    assert report.mean_agreement == 1.0
    assert report.is_conflicted is False
    assert report.most_conflicted_pair is None


def test_full_conflict_detected():
    """Build brief's own example: financial bullish, news bearish."""
    directions = {"financial": 0.8, "text": -0.7}
    report = compute_agreement(directions)
    assert report.mean_agreement == -1.0
    assert report.is_conflicted is True
    assert report.most_conflicted_pair == ("financial", "text")


def test_mixed_agreement_partial_conflict():
    directions = {"financial": 0.8, "text": 0.7, "microstructure": -0.6}
    report = compute_agreement(directions)
    # financial-text agree (+1), financial-micro disagree (-1), text-micro disagree (-1)
    assert report.mean_agreement == (1.0 - 1.0 - 1.0) / 3
    assert report.is_conflicted is True


def test_near_zero_signals_treated_as_neutral_not_forced_direction():
    directions = {"financial": 0.01, "text": -0.9}  # financial is within the neutral band
    report = compute_agreement(directions, neutral_band=0.05)
    assert report.pairwise_agreement[("financial", "text")] == 0.0


def test_conflict_threshold_is_configurable():
    directions = {"financial": 0.8, "text": 0.1}  # partial disagreement in magnitude, same sign
    report_strict = compute_agreement(directions, conflict_threshold=0.5)
    report_loose = compute_agreement(directions, conflict_threshold=-0.5)
    # same-sign directions -> agreement +1.0 regardless of threshold, so neither should be conflicted
    assert report_strict.is_conflicted is False
    assert report_loose.is_conflicted is False
