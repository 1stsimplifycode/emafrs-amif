import numpy as np
import pandas as pd
import pytest

from emafrs.models.labels import build_ternary_labels, drop_undefined_labels, forward_return


def test_forward_return_basic():
    close = pd.Series([100.0, 110.0, 99.0, 99.0])
    fwd = forward_return(close, horizon=1)
    assert fwd.iloc[0] == pytest.approx(0.10)
    assert fwd.iloc[1] == pytest.approx(99 / 110 - 1)
    assert np.isnan(fwd.iloc[-1])


def test_ternary_labels_respect_deadzone():
    close = pd.Series([100.0, 100.3, 99.7, 100.0])  # +0.3%, -0.6%, +0.3% forward moves
    labels = build_ternary_labels(close, horizon=1, deadzone=0.0025)
    assert labels.iloc[0] == 1   # +0.3% > 0.25% deadzone -> bullish
    assert labels.iloc[1] == -1  # -0.6% < -0.25% -> bearish
    assert np.isnan(labels.iloc[-1])  # no forward price


def test_ternary_labels_neutral_within_deadzone():
    close = pd.Series([100.0, 100.1])  # +0.1% forward move, inside 0.25% deadzone
    labels = build_ternary_labels(close, horizon=1, deadzone=0.0025)
    assert labels.iloc[0] == 0


def test_drop_undefined_labels_removes_trailing_and_warmup_rows():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    features = pd.DataFrame({"f1": [np.nan, 1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    labels = pd.Series([0.0, 1.0, -1.0, 0.0, 1.0, np.nan], index=idx)  # last is undefined
    X, y = drop_undefined_labels(features, labels)
    assert len(X) == 4  # drops row 0 (feature NaN) and row 5 (label NaN)
    assert not X.isna().any().any()
    assert not y.isna().any()
