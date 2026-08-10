"""Extended 5-signal manipulation risk model -- PDF Eq. 12 / Section 6.6.5:

    Risk_t = w1*S_t + w2*OBI_t + w3*Vol_anomaly + w4*OFI_t + w5*LS_t

HONESTY NOTE: the PDF states these weights "are learned during training." No labeled
historical pump-and-dump dataset currently exists in this reconstruction (building one
reliably -- e.g. from SEC litigation releases -- is nontrivial and deferred to
docs/EXPERIMENT_PROTOCOL.md / task #10's case studies), so `fit_weights` below is provided
for when labels exist, but the DEFAULT weighting is an explicitly disclosed equal-weight
prior, not a fitted value. This must never be presented as "the learned/optimal weights."

All five input signals must already be on a comparable scale (e.g. rolling z-scored / in
[-1,1]) before being combined here -- this module does not itself rescale them.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd


@dataclass
class ManipulationRiskWeights:
    w_hype: float
    w_obi: float
    w_volume_anomaly: float
    w_ofi: float
    w_liquidity_shock: float

    def as_array(self) -> np.ndarray:
        return np.array([self.w_hype, self.w_obi, self.w_volume_anomaly, self.w_ofi, self.w_liquidity_shock])


DEFAULT_WEIGHTS = ManipulationRiskWeights(
    w_hype=0.2, w_obi=0.2, w_volume_anomaly=0.2, w_ofi=0.2, w_liquidity_shock=0.2
)
"""Disclosed equal-weight prior -- NOT learned from labeled manipulation-event data. See
module docstring and docs/LIMITATIONS.md."""

_SIGNAL_COLUMNS = ["hype", "obi", "volume_anomaly", "ofi", "liquidity_shock"]


def regime_modulated_factor(
    regime: pd.Series, n_regimes: int = 4, microstructure_boost: float = 1.5
) -> pd.Series:
    """Linearly scales microstructure-signal weights up as regime severity increases
    (regime 1 -> factor 1.0, regime n_regimes -> factor `microstructure_boost`), matching
    PDF Sec 6.6.5's qualitative design intent that "microstructure weights are elevated
    during manipulation-risk regimes." `microstructure_boost` is a disclosed, fixed
    hyperparameter, not fit from data."""
    return 1.0 + (regime.astype(float) - 1.0) / (n_regimes - 1) * (microstructure_boost - 1.0)


def compute_manipulation_risk(
    signals: pd.DataFrame,
    regime: pd.Series,
    weights: ManipulationRiskWeights = DEFAULT_WEIGHTS,
    n_regimes: int = 4,
    microstructure_boost: float = 1.5,
) -> pd.Series:
    """signals must have columns hype, obi, volume_anomaly, ofi, liquidity_shock, all
    pre-scaled to a comparable range. Returns a per-row risk score with regime-modulated
    microstructure weighting, re-normalized so weights always sum to 1 at every row."""
    missing = set(_SIGNAL_COLUMNS) - set(signals.columns)
    if missing:
        raise ValueError(f"signals is missing required columns: {missing}")

    factor = regime_modulated_factor(regime.reindex(signals.index), n_regimes, microstructure_boost)

    w_hype = pd.Series(weights.w_hype, index=signals.index)
    w_obi = weights.w_obi * factor
    w_vol = weights.w_volume_anomaly * factor
    w_ofi = weights.w_ofi * factor
    w_ls = weights.w_liquidity_shock * factor
    total = w_hype + w_obi + w_vol + w_ofi + w_ls

    risk = (
        w_hype * signals["hype"] + w_obi * signals["obi"] + w_vol * signals["volume_anomaly"]
        + w_ofi * signals["ofi"] + w_ls * signals["liquidity_shock"]
    ) / total
    return risk.rename("manipulation_risk")


def rolling_flag_threshold(risk: pd.Series, window: int = 126, quantile: float = 0.95) -> pd.Series:
    """A causal (rolling, not full-sample) quantile threshold -- using a full-sample
    quantile would leak the future risk-score distribution into early-period alert
    decisions. tau_t is estimated only from risk[t-window+1:t+1]."""
    return risk.rolling(window, min_periods=window).quantile(quantile)


def flag_manipulation(risk: pd.Series, threshold: pd.Series) -> pd.Series:
    """PDF: "When Risk_t > tau, the system flags possible pump-and-dump activity." """
    return risk > threshold


def fit_weights_from_labels(
    signals: pd.DataFrame, labels: pd.Series, l1_reg: float = 0.1
) -> ManipulationRiskWeights:
    """Fits non-negative, sum-to-one weights via a constrained logistic regression against
    binary manipulation-event labels (1 = confirmed/suspected manipulation event, 0 =
    normal). Provided for when a labeled dataset exists -- NOT called anywhere in this
    reconstruction's default pipeline, since no such labeled dataset currently exists (see
    docs/LIMITATIONS.md). Calling this with fabricated labels would itself be a
    fabrication; only call it with a real, documented label source."""
    from sklearn.linear_model import LogisticRegression

    X = signals[_SIGNAL_COLUMNS].to_numpy()
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=1.0 / max(l1_reg, 1e-6))
    clf.fit(X, labels.to_numpy())
    raw = np.clip(clf.coef_[0], a_min=0.0, a_max=None)
    if raw.sum() == 0:
        raw = np.ones(5)
    normalized = raw / raw.sum()
    return ManipulationRiskWeights(*normalized)
