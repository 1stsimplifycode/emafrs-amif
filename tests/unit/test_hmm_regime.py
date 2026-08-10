"""Tests for the Market Regime Detection Engine against a synthetic series with KNOWN
ground-truth volatility regimes (alternating calm/turbulent blocks), so we can check the
HMM actually recovers something close to the true structure, not just that it runs."""
import numpy as np
import pandas as pd
import pytest

from emafrs.regime.evaluation import evaluate_regime_quality
from emafrs.regime.hmm_regime import build_regime_features, fit_regime_hmm


def _synthetic_two_regime_returns(n_per_block: int = 150, n_blocks: int = 6, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_per_block * n_blocks, freq="B")
    chunks = []
    for i in range(n_blocks):
        vol = 0.005 if i % 2 == 0 else 0.04  # calm vs turbulent, alternating
        chunks.append(rng.normal(0, vol, n_per_block))
    returns = np.concatenate(chunks)
    return pd.Series(returns, index=dates)


def test_regime_features_have_expected_columns_and_leakage_safe_nan_prefix():
    returns = _synthetic_two_regime_returns()
    feats = build_regime_features(returns, vol_window=21, entropy_window=21, hurst_window=126)
    assert {"realized_vol", "entropy", "hurst"}.issubset(feats.columns)
    assert feats["realized_vol"].iloc[:20].isna().all()


def test_hmm_recovers_two_dominant_volatility_regimes():
    returns = _synthetic_two_regime_returns()
    feats = build_regime_features(returns, vol_window=21, entropy_window=21, hurst_window=63)
    feats_reduced = feats[["realized_vol", "entropy"]]  # skip Hurst here: needs long window, adds NaN warmup

    model, labels = fit_regime_hmm(feats_reduced, vol_column="realized_vol", n_states=2, random_state=0)

    aligned_vol = feats_reduced.loc[labels.index, "realized_vol"]
    regime1_vol = aligned_vol[labels == 1].mean()
    regime2_vol = aligned_vol[labels == 2].mean()
    # by construction, regime 1 must be the lower-volatility state after relabeling
    assert regime1_vol < regime2_vol

    # the model should assign the turbulent blocks predominantly to the higher regime
    # (loose check: overall correlation between assigned regime and realized vol is positive)
    corr = aligned_vol.corr(labels.astype(float))
    assert corr > 0.3


def test_predict_on_new_data_uses_fitted_standardization():
    returns = _synthetic_two_regime_returns()
    feats = build_regime_features(returns, vol_window=21, entropy_window=21, hurst_window=63)[["realized_vol", "entropy"]]
    model, labels = fit_regime_hmm(feats, vol_column="realized_vol", n_states=2, random_state=0)

    predicted = model.predict(feats)
    assert set(predicted.unique()).issubset({1, 2})
    assert len(predicted) == len(feats.dropna())


def test_predict_proba_causal_matches_smoothed_at_final_row():
    """At the LAST row of any given sequence, filtered (causal) and smoothed posteriors
    must agree exactly -- there is no future observation left for smoothing to condition
    on beyond what filtering already used, so the two algorithms have identical
    information at that one point. A real equivalence check, not just a shape/type check."""
    returns = _synthetic_two_regime_returns()
    feats = build_regime_features(returns, vol_window=21, entropy_window=21, hurst_window=63)[["realized_vol", "entropy"]]
    model, _ = fit_regime_hmm(feats, vol_column="realized_vol", n_states=2, random_state=0)

    smoothed = model.predict_proba(feats)
    causal = model.predict_proba_causal(feats)
    last_idx = smoothed.index[-1]
    np.testing.assert_allclose(
        smoothed.loc[last_idx].to_numpy(), causal.loc[last_idx].to_numpy(), atol=1e-8
    )


def test_predict_proba_causal_is_truncation_invariant():
    """The core leakage-safety property, proven directly via the same truncation
    methodology `assert_no_lookahead` uses elsewhere in this codebase: a date's CAUSAL
    regime probability must be identical whether or not future rows are appended to the
    input, since it must never depend on them. hmmlearn's own (smoothed) `predict_proba`
    does NOT have this property in general -- this is exactly the leakage pathway
    diagnosed in docs/DATA_LEAKAGE_AUDIT.md, distinct from and not caught by any
    per-feature no-lookahead check, since it lives in how the already-fit HMM is APPLIED,
    not in how its input features are computed."""
    returns = _synthetic_two_regime_returns(n_per_block=150, n_blocks=8)
    feats = build_regime_features(returns, vol_window=21, entropy_window=21, hurst_window=63)[["realized_vol", "entropy"]]
    clean = feats.dropna()

    fit_window = clean.iloc[:400]
    model, _ = fit_regime_hmm(fit_window, vol_column="realized_vol", n_states=2, random_state=0)

    full_causal = model.predict_proba_causal(clean)
    truncated_causal = model.predict_proba_causal(clean.iloc[:700])

    common = full_causal.index.intersection(truncated_causal.index)
    safe_idx = common[common < clean.index[600]]  # margin before the truncation point
    assert len(safe_idx) > 50
    diff = (full_causal.loc[safe_idx] - truncated_causal.loc[safe_idx]).abs().max().max()
    assert diff < 1e-10


def test_fit_raises_on_insufficient_data():
    tiny = pd.DataFrame({"realized_vol": [0.1] * 10, "entropy": [1.0] * 10})
    with pytest.raises(ValueError):
        fit_regime_hmm(tiny, vol_column="realized_vol", n_states=4)


def test_evaluate_regime_quality_flags_reasonable_and_bad_cases():
    returns = _synthetic_two_regime_returns()
    feats = build_regime_features(returns, vol_window=21, entropy_window=21, hurst_window=63)[["realized_vol", "entropy"]]
    model, labels = fit_regime_hmm(feats, vol_column="realized_vol", n_states=2, random_state=0)

    report = evaluate_regime_quality(
        labels, feats, vol_column="realized_vol", transition_matrix=model.model.transmat_
    )
    assert report.monotonic_vol_ordering is True
    assert report.n_states == 2
    assert report.mean_run_length > 0

    # a degenerate all-one-label series should trip the "not well separated" / run-length logic
    degenerate_labels = pd.Series(1, index=labels.index, name="regime")
    degenerate_report = evaluate_regime_quality(
        degenerate_labels, feats, vol_column="realized_vol", transition_matrix=model.model.transmat_
    )
    assert degenerate_report.n_states == 1
