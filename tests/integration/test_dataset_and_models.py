"""Integration tests: full pipeline (real yfinance data -> dataset -> baselines ->
fusion models) end to end. Slow (network + HMM fit + NN training), marked integration."""
import numpy as np
import pytest

from emafrs.features.leakage_audit import check_chronological_split
from emafrs.models.baselines import ModelAOpaqueXGBoost, ModelBExplainableLogReg, encode_labels, evaluate_predictions
from emafrs.models.dataset import build_daily_dataset
from emafrs.models.train import train_fusion_model

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def bundle():
    return build_daily_dataset("SPY", period="3y", horizon=1, deadzone=0.0025)


def test_dataset_bundle_structure(bundle):
    assert len(bundle.train_idx) > 0 and len(bundle.val_idx) > 0 and len(bundle.test_idx) > 0
    assert set(bundle.labels.unique()).issubset({-1, 0, 1})
    assert bundle.regime_columns  # regime posterior columns present
    assert not bundle.features.isna().any().any()  # all NaN rows were dropped


def test_dataset_split_is_chronological_no_overlap(bundle):
    train_result = check_chronological_split(bundle.train_idx, bundle.val_idx)
    val_result = check_chronological_split(bundle.val_idx, bundle.test_idx)
    assert train_result.passed, train_result.details
    assert val_result.passed, val_result.details


def test_regime_model_fit_only_on_train_split(bundle):
    """The regime HMM must never have seen validation/test dates during fitting.

    NOTE: build_daily_dataset() splits the regime feature index BEFORE joining with
    financial features (whose NaN warm-up profile differs -- e.g. SMA50 needs 50 rows,
    the regime block's Hurst window needs 126), so the regime-fit train boundary is not
    byte-identical to `bundle.train_idx`'s boundary; both are valid, chronologically-first
    prefixes of their own respective indices. The invariant this test actually needs to
    verify is the leakage-relevant one: the regime model's fitted standardization
    parameters must match a mean/std computed from SOME prefix of regime_input that ends
    at or before bundle.train_idx's own end (never extending into val/test dates) -- which
    is exactly how dataset.py derives it. Re-derive it the same way and compare exactly.
    """
    from emafrs.regime.hmm_regime import build_regime_features
    from emafrs.data.yfinance_client import YFinanceClient
    from emafrs.features.technical import compute_returns

    df, _ = YFinanceClient().get_ohlcv("SPY", period="3y", interval="1d")
    returns = compute_returns(df["close"])
    regime_input = build_regime_features(returns, volume=df["volume"])
    dates_with_regime_features = regime_input.dropna().index

    n = len(dates_with_regime_features)
    train_end_idx = int(n * 0.7)  # mirrors dataset.py's _chronological_split(train_frac=0.7)
    train_dates_for_regime = dates_with_regime_features[:train_end_idx]

    assert train_dates_for_regime.max() <= bundle.train_idx.max()  # never reaches into val/test

    expected_mean = regime_input.loc[train_dates_for_regime].mean()
    for col in bundle.regime_model.feature_mean.index:
        assert bundle.regime_model.feature_mean[col] == pytest.approx(expected_mean[col], rel=1e-6)


def test_baseline_models_fit_and_predict(bundle):
    X_train = bundle.features.loc[bundle.train_idx, bundle.financial_columns]
    X_test = bundle.features.loc[bundle.test_idx, bundle.financial_columns]
    y_train = bundle.labels.loc[bundle.train_idx]
    y_test = bundle.labels.loc[bundle.test_idx]

    model_a = ModelAOpaqueXGBoost(n_estimators=50).fit(X_train, y_train)
    proba_a = model_a.predict_proba(X_test)
    assert proba_a.shape == (len(X_test), 3)
    assert np.allclose(proba_a.sum(axis=1), 1.0, atol=1e-5)

    model_b = ModelBExplainableLogReg().fit(X_train, y_train)
    proba_b = model_b.predict_proba(X_test)
    assert proba_b.shape == (len(X_test), 3)
    coefs = model_b.coefficients(bundle.financial_columns)
    assert coefs.shape == (3, len(bundle.financial_columns))

    result = evaluate_predictions("test", encode_labels(y_test), proba_a.argmax(axis=1), proba_a)
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.macro_f1 <= 1.0


def test_fusion_models_train_and_produce_valid_weights(bundle):
    modality_cols = {"financial": bundle.financial_columns}
    if len(bundle.microstructure_columns) > 1:
        modality_cols["microstructure"] = bundle.microstructure_columns

    _, result_static, weights_static = train_fusion_model(
        bundle, "static", modality_cols, epochs=20, patience=5
    )
    _, result_rama, weights_rama = train_fusion_model(
        bundle, "rama", modality_cols, epochs=20, patience=5
    )

    for weights in (weights_static, weights_rama):
        assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-4)
        assert (weights >= 0).all()

    assert 0.0 <= result_static.accuracy <= 1.0
    assert 0.0 <= result_rama.accuracy <= 1.0
