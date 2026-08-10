import numpy as np
import pandas as pd

from emafrs.mlops.drift import detect_feature_drift


def test_no_drift_when_distributions_match():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"f1": rng.normal(0, 1, 500), "f2": rng.normal(5, 2, 500)})
    recent = pd.DataFrame({"f1": rng.normal(0, 1, 200), "f2": rng.normal(5, 2, 200)})
    report = detect_feature_drift(reference, recent)
    assert report.drift_fraction < 0.5
    assert report.retrain_recommended is False


def test_drift_detected_when_distribution_shifts():
    rng = np.random.default_rng(1)
    reference = pd.DataFrame({"f1": rng.normal(0, 1, 500), "f2": rng.normal(5, 2, 500)})
    recent = pd.DataFrame({"f1": rng.normal(10, 1, 200), "f2": rng.normal(5, 2, 200)})  # f1 shifted hard
    report = detect_feature_drift(reference, recent)
    assert "f1" in report.drifted_features
    assert report.feature_drift_pvalues["f1"] < 0.01


def test_retrain_recommended_when_many_features_drift():
    rng = np.random.default_rng(2)
    reference = pd.DataFrame({f"f{i}": rng.normal(0, 1, 300) for i in range(10)})
    recent = pd.DataFrame({f"f{i}": rng.normal(8, 1, 150) for i in range(10)})  # all shifted
    report = detect_feature_drift(reference, recent, drift_fraction_threshold=0.2)
    assert report.drift_fraction > 0.2
    assert report.retrain_recommended is True


def test_skips_columns_with_too_few_observations():
    reference = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})  # only 3 obs, below the min-10 threshold
    recent = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    report = detect_feature_drift(reference, recent)
    assert "f1" not in report.feature_drift_pvalues


def test_ignores_columns_not_present_in_recent():
    rng = np.random.default_rng(3)
    reference = pd.DataFrame({"f1": rng.normal(0, 1, 50), "f2": rng.normal(0, 1, 50)})
    recent = pd.DataFrame({"f1": rng.normal(0, 1, 50)})  # f2 missing
    report = detect_feature_drift(reference, recent)
    assert "f2" not in report.feature_drift_pvalues
    assert "f1" in report.feature_drift_pvalues
