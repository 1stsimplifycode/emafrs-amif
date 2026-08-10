import numpy as np
import pandas as pd

from emafrs.features.preprocessing import align_to_business_calendar, rolling_zscore


def test_rolling_zscore_is_roughly_standard_normal_on_stationary_noise():
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(0, 1, 500))
    z = rolling_zscore(series, window=100).dropna()
    assert abs(z.mean()) < 0.3
    assert abs(z.std() - 1.0) < 0.3


def test_rolling_zscore_early_values_are_nan_until_window_fills():
    series = pd.Series(np.arange(50, dtype=float))
    z = rolling_zscore(series, window=20)
    assert z.iloc[:19].isna().all()
    assert z.iloc[19:].notna().all()


def test_align_to_business_calendar_forward_fills_gaps_only_from_past():
    dates = pd.to_datetime(["2023-01-02", "2023-01-04"])  # gap on 2023-01-03
    df = pd.DataFrame({"value": [10.0, 20.0]}, index=dates)
    aligned = align_to_business_calendar(df, freq="D")
    assert aligned.loc["2023-01-03", "value"] == 10.0  # filled from the PAST value only
