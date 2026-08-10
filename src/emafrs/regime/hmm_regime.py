"""Market Regime Detection Engine (MRDE): Gaussian HMM over realized volatility, Shannon
entropy, and Hurst exponent (PDF Section 6.5), fit via Baum-Welch (hmmlearn's default EM).

hmmlearn's hidden state indices are arbitrary (state 0 is not necessarily "low volatility").
This module relabels states 1..n_states by ascending mean realized volatility so the output
regime index is always interpretable as Low(1) < Medium(2) < High(3) < Extreme(4), matching
the PDF's R_t in {1,2,3,4}.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

REGIME_NAMES = {1: "Low", 2: "Medium", 3: "High", 4: "Extreme"}


@dataclass
class RegimeModel:
    model: GaussianHMM
    feature_columns: list[str]
    vol_column: str
    state_to_regime: dict[int, int]
    state_means: pd.DataFrame  # indexed by regime (1..n_states), columns = feature_columns
    feature_mean: pd.Series  # standardization params used at fit time
    feature_std: pd.Series

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict regime labels for new feature rows using the already-fit HMM.
        Rows containing NaN in any required feature column are dropped (regime undefined
        until enough history has accumulated), never imputed."""
        clean = features[self.feature_columns].dropna()
        if clean.empty:
            return pd.Series(dtype=int, name="regime")
        x_std = (clean - self.feature_mean) / self.feature_std
        hidden_states = self.model.predict(x_std.values)
        regimes = pd.Series(
            [self.state_to_regime[s] for s in hidden_states], index=clean.index, name="regime"
        )
        return regimes

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Posterior state probabilities, columns relabeled to regime order 1..n_states --
        used for the epistemic-uncertainty-aware temperature scaling in the VAFE (PDF Sec 6.7.3)."""
        clean = features[self.feature_columns].dropna()
        if clean.empty:
            return pd.DataFrame(columns=sorted(self.state_to_regime.values()))
        x_std = (clean - self.feature_mean) / self.feature_std
        posteriors = self.model.predict_proba(x_std.values)
        regime_order = [self.state_to_regime[s] for s in range(posteriors.shape[1])]
        df = pd.DataFrame(posteriors, index=clean.index, columns=regime_order)
        return df[sorted(df.columns)]


def fit_regime_hmm(
    features: pd.DataFrame,
    vol_column: str,
    n_states: int = 4,
    n_iter: int = 200,
    random_state: int = 42,
) -> tuple[RegimeModel, pd.Series]:
    """features: DataFrame of regime-relevant columns (realized vol, entropy, Hurst, ...),
    already computed via rolling windows so it is leakage-safe by construction (see
    info_dynamics.py and features/technical.py). Rows with any NaN are dropped before
    fitting (early warm-up period). `vol_column` must be one of features.columns and is
    used only to order/relabel hidden states by ascending volatility -- it does not receive
    special weight in the HMM's emission model itself.

    Returns (fitted RegimeModel, in-sample regime labels aligned to the cleaned index).
    """
    if vol_column not in features.columns:
        raise ValueError(f"vol_column={vol_column!r} not in features.columns={list(features.columns)}")

    clean = features.dropna()
    if len(clean) < n_states * 20:
        raise ValueError(
            f"Not enough clean observations ({len(clean)}) to fit a {n_states}-state HMM reliably; "
            "need at least ~20 observations per state."
        )

    feature_mean = clean.mean()
    feature_std = clean.std().replace(0.0, 1.0)
    x_std = (clean - feature_mean) / feature_std

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(x_std.values)
    hidden_states = model.predict(x_std.values)

    vol_idx = list(features.columns).index(vol_column)
    state_vol_means = [x_std.values[hidden_states == s, vol_idx].mean() if (hidden_states == s).any() else np.inf
                        for s in range(n_states)]
    state_order = np.argsort(state_vol_means)  # ascending vol -> regime 1..n_states
    state_to_regime = {int(old): int(new) + 1 for new, old in enumerate(state_order)}

    regime_labels = pd.Series(
        [state_to_regime[s] for s in hidden_states], index=clean.index, name="regime"
    )

    means_unstd = model.means_ * feature_std.values + feature_mean.values
    state_means = pd.DataFrame(means_unstd, columns=features.columns)
    state_means.index = [state_to_regime[i] for i in range(n_states)]
    state_means = state_means.sort_index()

    regime_model = RegimeModel(
        model=model,
        feature_columns=list(features.columns),
        vol_column=vol_column,
        state_to_regime=state_to_regime,
        state_means=state_means,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    return regime_model, regime_labels


def build_regime_features(
    returns: pd.Series,
    volume: pd.Series | None = None,
    vol_window: int = 21,
    entropy_window: int = 21,
    hurst_window: int = 126,
) -> pd.DataFrame:
    """Assembles the MRDE's observable state variables from a returns series (+ optional
    volume), per PDF Section 6.5: realized volatility, Shannon entropy, Hurst exponent."""
    from emafrs.features.technical import compute_realized_volatility
    from emafrs.regime.info_dynamics import rolling_entropy, rolling_hurst

    out = pd.DataFrame(index=returns.index)
    out["realized_vol"] = compute_realized_volatility(returns, window=vol_window)
    out["entropy"] = rolling_entropy(returns, window=entropy_window)
    out["hurst"] = rolling_hurst(returns, window=hurst_window)
    if volume is not None:
        vol_mean = volume.rolling(vol_window, min_periods=vol_window).mean()
        vol_std = volume.rolling(vol_window, min_periods=vol_window).std()
        out["volume_zscore"] = (volume - vol_mean) / vol_std
    return out
