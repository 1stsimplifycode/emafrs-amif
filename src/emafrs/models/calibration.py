"""Regime-dependent calibration -- PDF Eq. 18:

    P(y_t | z_t, R_t) = softmax(f_theta(z_t) * gamma(R_t))

where gamma(R_t) is a temperature that decreases monotonically with regime severity
(more uncertainty attenuation in higher-volatility regimes). Also provides standard
calibration metrics (Brier score, Expected Calibration Error) used to check whether that
design intent actually holds empirically, per build brief Section 21 ("evaluate: ...
calibration").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def fit_regime_temperature(
    logits: torch.Tensor, labels: torch.Tensor, regime_idx: torch.Tensor, n_regimes: int,
    lr: float = 0.01, max_iter: int = 200,
) -> torch.Tensor:
    """Fits one temperature parameter per regime by minimizing NLL on a held-out
    (validation) set, standard post-hoc temperature scaling (Guo et al. 2017) applied
    per-regime rather than globally. Returns a (n_regimes,) tensor of temperatures > 0.

    Must be fit on a VALIDATION split disjoint from both training and the test split being
    calibrated for -- fitting on test data would leak test-set information into the
    calibration itself."""
    raw_temp = torch.zeros(n_regimes, requires_grad=True)  # temperature = softplus(raw) > 0
    optimizer = torch.optim.LBFGS([raw_temp], lr=lr, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        temp = torch.nn.functional.softplus(raw_temp) + 1e-3
        temp_per_row = temp[regime_idx]
        scaled = logits / temp_per_row.unsqueeze(1)
        loss = nll(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return torch.nn.functional.softplus(raw_temp) + 1e-3


def fit_global_temperature(logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 200) -> torch.Tensor:
    """Single temperature fit across ALL validation rows regardless of regime -- the
    n_regimes=1 special case, used as a baseline to isolate whether per-regime
    differentiation is actually earning its complexity (see the shrinkage variant below,
    which was added after `fit_regime_temperature` was found to make test-set calibration
    WORSE on a small (~40 rows/regime) validation split -- see
    docs/EXPERIMENT_PROTOCOL.md's Calibration section)."""
    raw_temp = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([raw_temp], lr=lr, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        temp = torch.nn.functional.softplus(raw_temp) + 1e-3
        loss = nll(logits / temp, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return torch.nn.functional.softplus(raw_temp) + 1e-3


def fit_regime_temperature_shrinkage(
    logits: torch.Tensor, labels: torch.Tensor, regime_idx: torch.Tensor, n_regimes: int,
    shrinkage_weight: float = 2.0, lr: float = 0.01, max_iter: int = 200,
) -> torch.Tensor:
    """Partial-pooling variant of `fit_regime_temperature`: per-regime temperatures are
    fit jointly with a single global temperature and penalized (L2) for deviating from
    it, i.e. shrunk toward the global value rather than left fully free. This directly
    targets the diagnosed root cause of the naive per-regime fit's test-set calibration
    regression: ~40 validation rows/regime is little enough data that 4 independent free
    temperature parameters can fit validation-split noise that does not generalize.
    `shrinkage_weight=0` recovers the naive independent-per-regime fit exactly;
    `shrinkage_weight -> inf` recovers a single global temperature for all regimes.
    `shrinkage_weight=2.0` is a fixed, disclosed default (not tuned against the test set
    this function's output is later evaluated on -- that would be calibration leakage)."""
    raw_temp = torch.zeros(n_regimes, requires_grad=True)
    raw_global = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([raw_temp, raw_global], lr=lr, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        temp = torch.nn.functional.softplus(raw_temp) + 1e-3
        global_temp = torch.nn.functional.softplus(raw_global) + 1e-3
        temp_per_row = temp[regime_idx]
        scaled = logits / temp_per_row.unsqueeze(1)
        loss = nll(scaled, labels) + shrinkage_weight * torch.mean((temp - global_temp) ** 2)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return torch.nn.functional.softplus(raw_temp) + 1e-3


def apply_regime_temperature(logits: torch.Tensor, regime_idx: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    temp_per_row = temperature[regime_idx]
    return torch.softmax(logits / temp_per_row.unsqueeze(1), dim=1)


def brier_score(proba: np.ndarray, y_true_idx: np.ndarray, n_classes: int = 3) -> float:
    """Multi-class Brier score: mean squared error between predicted probabilities and
    one-hot true labels, averaged over classes and samples. Lower is better calibrated."""
    one_hot = np.eye(n_classes)[y_true_idx]
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def expected_calibration_error(proba: np.ndarray, y_true_idx: np.ndarray, n_bins: int = 10) -> float:
    """ECE over the model's top-predicted-class confidence: bins predictions by confidence,
    compares mean confidence to empirical accuracy within each bin, weighted by bin size."""
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    correct = (predictions == y_true_idx).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true_idx)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


@dataclass
class RegimeCalibrationReport:
    per_regime_brier: dict[int, float]
    per_regime_ece: dict[int, float]
    overall_brier: float
    overall_ece: float
    temperature_by_regime: dict[int, float]


def evaluate_regime_calibration(
    proba: np.ndarray, y_true_idx: np.ndarray, regime_idx: np.ndarray, temperature: np.ndarray, n_classes: int = 3
) -> RegimeCalibrationReport:
    per_regime_brier, per_regime_ece = {}, {}
    for r in sorted(set(regime_idx.tolist())):
        mask = regime_idx == r
        if mask.sum() < 5:
            continue
        per_regime_brier[int(r)] = brier_score(proba[mask], y_true_idx[mask], n_classes)
        per_regime_ece[int(r)] = expected_calibration_error(proba[mask], y_true_idx[mask])

    return RegimeCalibrationReport(
        per_regime_brier=per_regime_brier,
        per_regime_ece=per_regime_ece,
        overall_brier=brier_score(proba, y_true_idx, n_classes),
        overall_ece=expected_calibration_error(proba, y_true_idx),
        temperature_by_regime={i + 1: float(t) for i, t in enumerate(temperature)},
    )
