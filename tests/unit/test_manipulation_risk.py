import numpy as np
import pandas as pd
import pytest

from emafrs.features.leakage_audit import assert_no_lookahead
from emafrs.risk.manipulation_risk import (
    DEFAULT_WEIGHTS,
    ManipulationRiskWeights,
    compute_manipulation_risk,
    flag_manipulation,
    regime_modulated_factor,
    rolling_flag_threshold,
)


def _signals_df(n=100, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "hype": rng.normal(0, 1, n),
            "obi": rng.normal(0, 1, n),
            "volume_anomaly": rng.normal(0, 1, n),
            "ofi": rng.normal(0, 1, n),
            "liquidity_shock": rng.normal(0, 1, n),
        },
        index=idx,
    )


def test_regime_modulated_factor_increases_with_regime_severity():
    regime = pd.Series([1, 2, 3, 4])
    factor = regime_modulated_factor(regime, n_regimes=4, microstructure_boost=1.5)
    assert factor.iloc[0] == pytest.approx(1.0)
    assert factor.iloc[-1] == pytest.approx(1.5)
    assert factor.is_monotonic_increasing


def test_compute_manipulation_risk_requires_all_five_signals():
    signals = _signals_df().drop(columns=["ofi"])
    regime = pd.Series(1, index=signals.index)
    with pytest.raises(ValueError):
        compute_manipulation_risk(signals, regime)


def test_compute_manipulation_risk_extreme_case_all_signals_agree():
    """When every signal is strongly positive, risk should be strongly positive; when all
    strongly negative, risk should be strongly negative -- basic sanity/direction check."""
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    high_signals = pd.DataFrame(
        {"hype": [3.0, -3.0], "obi": [3.0, -3.0], "volume_anomaly": [3.0, -3.0],
         "ofi": [3.0, -3.0], "liquidity_shock": [3.0, -3.0]}, index=idx,
    )
    regime = pd.Series([1, 1], index=idx)
    risk = compute_manipulation_risk(high_signals, regime)
    assert risk.iloc[0] > 0
    assert risk.iloc[1] < 0
    assert risk.iloc[0] == pytest.approx(3.0)  # equal weights averaging identical inputs


def test_regime_modulation_elevates_microstructure_over_hype_at_high_regime():
    """At high regime severity, a strong microstructure-only signal should score higher
    risk than an equally-strong hype-only signal, reflecting the elevated microstructure
    weighting design intent (PDF Sec 6.6.5)."""
    idx = pd.date_range("2024-01-01", periods=1, freq="D")
    hype_only = pd.DataFrame({"hype": [3.0], "obi": [0.0], "volume_anomaly": [0.0], "ofi": [0.0], "liquidity_shock": [0.0]}, index=idx)
    micro_only = pd.DataFrame({"hype": [0.0], "obi": [3.0], "volume_anomaly": [0.0], "ofi": [0.0], "liquidity_shock": [0.0]}, index=idx)
    regime_high = pd.Series([4], index=idx)

    risk_hype = compute_manipulation_risk(hype_only, regime_high, n_regimes=4, microstructure_boost=2.0)
    risk_micro = compute_manipulation_risk(micro_only, regime_high, n_regimes=4, microstructure_boost=2.0)
    assert risk_micro.iloc[0] > risk_hype.iloc[0]


def test_rolling_flag_threshold_and_flag_manipulation_use_only_past_data():
    signals = _signals_df(n=200)
    regime = pd.Series(np.random.default_rng(1).integers(1, 5, len(signals)), index=signals.index)
    risk = compute_manipulation_risk(signals, regime)

    result = assert_no_lookahead(
        lambda d: rolling_flag_threshold(compute_manipulation_risk(
            d[["hype", "obi", "volume_anomaly", "ofi", "liquidity_shock"]],
            regime.reindex(d.index), n_regimes=4,
        ), window=50, quantile=0.9),
        signals, n_samples=8, min_index=60,
    )
    assert result.passed, result.details

    threshold = rolling_flag_threshold(risk, window=50, quantile=0.9)
    flags = flag_manipulation(risk, threshold)
    assert flags.dtype == bool
    # roughly ~10% of post-warmup rows should be flagged given a 90th-percentile threshold
    valid = flags.loc[threshold.notna()]
    assert 0.0 < valid.mean() < 0.3


def test_default_weights_sum_to_one():
    total = DEFAULT_WEIGHTS.w_hype + DEFAULT_WEIGHTS.w_obi + DEFAULT_WEIGHTS.w_volume_anomaly \
        + DEFAULT_WEIGHTS.w_ofi + DEFAULT_WEIGHTS.w_liquidity_shock
    assert total == pytest.approx(1.0)
