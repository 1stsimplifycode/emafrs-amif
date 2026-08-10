from emafrs.risk.cross_modal_conflict import compute_agreement


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
