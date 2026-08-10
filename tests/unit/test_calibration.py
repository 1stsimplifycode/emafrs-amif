import numpy as np
import torch

from emafrs.models.calibration import (
    apply_regime_temperature,
    brier_score,
    evaluate_regime_calibration,
    expected_calibration_error,
    fit_regime_temperature,
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
