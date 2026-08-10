"""Market Regime Detection Engine (MRDE) -- PDF Section 6.5.

fit_regime_hmm() fits a Gaussian HMM (Baum-Welch) over realized volatility, Shannon
entropy, and Hurst exponent, relabeling hidden states 1..n_states by ascending mean
volatility so regime 1=Low ... 4=Extreme is always interpretable. evaluate_regime_quality()
checks persistence, separation, and vol-ordering rather than assuming the fit is meaningful.
"""
from emafrs.regime.hmm_regime import REGIME_NAMES, RegimeModel, build_regime_features, fit_regime_hmm

__all__ = ["REGIME_NAMES", "RegimeModel", "build_regime_features", "fit_regime_hmm"]
