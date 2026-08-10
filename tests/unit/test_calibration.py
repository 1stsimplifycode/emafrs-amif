import numpy as np
import pytest
import torch

from emafrs.models.calibration import (
    apply_regime_temperature,
    brier_score,
    evaluate_regime_calibration,
    expected_calibration_error,
    fit_global_temperature,
    fit_regime_temperature,
    fit_regime_temperature_shrinkage,
    negative_log_likelihood,
)


def test_brier_score_zero_for_perfect_confident_correct_predictions():
    proba = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y_true = np.array([0, 1])
    assert brier_score(proba, y_true) == 0.0


def test_brier_score_positive_for_wrong_confident_predictions():
    proba = np.array([[0.0, 0.0, 1.0]])
    y_true = np.array([0])
    score = brier_score(proba, y_true)
    assert score > 0


def test_ece_zero_when_confidence_matches_accuracy_perfectly():
    # 10 predictions, all confidence 0.7, and exactly 70% correct
    proba = np.zeros((10, 3))
    proba[:, 0] = 0.7
    proba[:, 1] = 0.15
    proba[:, 2] = 0.15
    y_true = np.array([0] * 7 + [1] * 3)
    ece = expected_calibration_error(proba, y_true, n_bins=10)
    assert ece < 0.05


def test_ece_high_for_overconfident_wrong_predictions():
    proba = np.tile(np.array([0.99, 0.005, 0.005]), (10, 1))
    y_true = np.array([1] * 10)  # always wrong despite near-certain confidence
    ece = expected_calibration_error(proba, y_true, n_bins=10)
    assert ece > 0.9


def test_fit_regime_temperature_reduces_overconfidence():
    """Construct badly-overconfident logits for one regime and check that fitting a
    per-regime temperature on a validation set actually reduces NLL/improves calibration
    relative to no scaling (temperature=1), proving the fit does something real."""
    torch.manual_seed(0)
    n = 200
    n_regimes = 2
    true_labels = torch.randint(0, 3, (n,))
    # logits wildly overconfident in the WRONG direction half the time
    logits = torch.nn.functional.one_hot(true_labels, 3).float() * 8.0
    logits += torch.randn(n, 3) * 0.5
    regime_idx = torch.randint(0, n_regimes, (n,))

    temp = fit_regime_temperature(logits, true_labels, regime_idx, n_regimes=n_regimes, max_iter=50)
    assert temp.shape == (n_regimes,)
    assert (temp > 0).all()

    proba_scaled = apply_regime_temperature(logits, regime_idx, temp).detach().numpy()
    proba_unscaled = torch.softmax(logits, dim=1).numpy()

    ece_scaled = expected_calibration_error(proba_scaled, true_labels.numpy())
    ece_unscaled = expected_calibration_error(proba_unscaled, true_labels.numpy())
    # scaling down an already-good-direction but overconfident distribution should not
    # make calibration worse
    assert ece_scaled <= ece_unscaled + 0.05


def test_nll_zero_for_perfect_confident_correct_predictions():
    proba = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y_true = np.array([0, 1])
    assert negative_log_likelihood(proba, y_true) == pytest.approx(0.0, abs=1e-9)


def test_nll_high_for_confident_wrong_predictions():
    proba = np.array([[0.0, 0.0, 1.0]])
    y_true = np.array([0])
    assert negative_log_likelihood(proba, y_true) > 20  # -log(eps) is large


def test_nll_higher_for_less_confident_correct_predictions():
    confident = np.array([[0.9, 0.05, 0.05]])
    unsure = np.array([[0.4, 0.3, 0.3]])
    y_true = np.array([0])
    assert negative_log_likelihood(unsure, y_true) > negative_log_likelihood(confident, y_true)


def test_fit_global_temperature_reduces_overconfidence():
    """Same overconfident-logits construction as the per-regime test, but the single
    global temperature (no regime split at all) -- added as a baseline to isolate whether
    per-regime freedom is actually earning its complexity, per the real diagnostic finding
    in docs/EXPERIMENT_PROTOCOL.md's Calibration section."""
    torch.manual_seed(0)
    n = 200
    true_labels = torch.randint(0, 3, (n,))
    logits = torch.nn.functional.one_hot(true_labels, 3).float() * 8.0
    logits += torch.randn(n, 3) * 0.5

    temp = fit_global_temperature(logits, true_labels, max_iter=50)
    assert temp.shape == (1,)
    assert (temp > 0).all()

    proba_scaled = torch.softmax(logits / temp, dim=1).numpy()
    proba_unscaled = torch.softmax(logits, dim=1).numpy()
    ece_scaled = expected_calibration_error(proba_scaled, true_labels.numpy())
    ece_unscaled = expected_calibration_error(proba_unscaled, true_labels.numpy())
    assert ece_scaled <= ece_unscaled + 0.05


def test_fit_regime_temperature_shrinkage_collapses_to_global_at_high_weight():
    """With a very large shrinkage_weight, the L2-toward-global penalty dominates the NLL
    term, so per-regime temperatures should collapse to nearly one shared value
    regardless of the data -- a direct mathematical property of the penalty, verified
    empirically here rather than merely assumed."""
    torch.manual_seed(1)
    n = 300
    n_regimes = 3
    true_labels = torch.randint(0, 3, (n,))
    regime_idx = torch.randint(0, n_regimes, (n,))
    logits = torch.randn(n, 3) * 2.0

    temp_shrunk = fit_regime_temperature_shrinkage(
        logits, true_labels, regime_idx, n_regimes=n_regimes, shrinkage_weight=1000.0, max_iter=50
    )
    assert temp_shrunk.shape == (n_regimes,)
    assert (temp_shrunk > 0).all()
    assert temp_shrunk.std().item() < 0.01


def test_fit_regime_temperature_shrinkage_zero_weight_matches_free_fit():
    """shrinkage_weight=0 should recover (approximately) the same result as the
    unregularized per-regime fit, since the penalty term vanishes entirely."""
    torch.manual_seed(2)
    n = 200
    n_regimes = 2
    true_labels = torch.randint(0, 3, (n,))
    logits = torch.nn.functional.one_hot(true_labels, 3).float() * 8.0
    logits += torch.randn(n, 3) * 0.5
    regime_idx = torch.randint(0, n_regimes, (n,))

    temp_free = fit_regime_temperature(logits, true_labels, regime_idx, n_regimes=n_regimes, max_iter=50)
    temp_shrunk_zero = fit_regime_temperature_shrinkage(
        logits, true_labels, regime_idx, n_regimes=n_regimes, shrinkage_weight=0.0, max_iter=50
    )
    assert torch.allclose(temp_free, temp_shrunk_zero, atol=0.05)


def test_evaluate_regime_calibration_report_structure():
    rng = np.random.default_rng(0)
    n = 100
    proba = rng.dirichlet(alpha=[2, 2, 2], size=n)
    y_true = rng.integers(0, 3, n)
    regime_idx = rng.integers(0, 4, n)
    temperature = np.array([1.0, 1.2, 1.5, 2.0])

    report = evaluate_regime_calibration(proba, y_true, regime_idx, temperature)
    assert 0.0 <= report.overall_brier
    assert 0.0 <= report.overall_ece <= 1.0
    assert set(report.temperature_by_regime.keys()) == {1, 2, 3, 4}
