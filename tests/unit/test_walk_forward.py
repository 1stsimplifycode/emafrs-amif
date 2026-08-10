import pandas as pd
import pytest

from emafrs.experiments.walk_forward import generate_walk_forward_folds
from emafrs.features.leakage_audit import check_chronological_split


def _dates(n=1000):
    return pd.date_range("2018-01-01", periods=n, freq="B")


def test_folds_are_chronologically_ordered_within_each_fold():
    folds = generate_walk_forward_folds(_dates(), n_folds=4, initial_train_frac=0.5, val_frac=0.05)
    assert len(folds) == 4
    for fold in folds:
        r1 = check_chronological_split(fold.train_idx, fold.val_idx)
        r2 = check_chronological_split(fold.val_idx, fold.test_idx)
        assert r1.passed, (fold.fold_id, r1.details)
        assert r2.passed, (fold.fold_id, r2.details)


def test_test_windows_across_folds_are_non_overlapping_and_increasing():
    folds = generate_walk_forward_folds(_dates(), n_folds=4, initial_train_frac=0.5, val_frac=0.05)
    for i in range(len(folds) - 1):
        assert folds[i].test_idx.max() < folds[i + 1].test_idx.min()
        assert not set(folds[i].test_idx) & set(folds[i + 1].test_idx)


def test_expanding_window_training_set_grows_across_folds():
    folds = generate_walk_forward_folds(_dates(), n_folds=4, initial_train_frac=0.5, val_frac=0.05, expanding=True)
    sizes = [len(f.train_idx) for f in folds]
    assert sizes == sorted(sizes)  # non-decreasing
    assert sizes[-1] > sizes[0]


def test_rolling_window_training_set_size_bounded():
    folds = generate_walk_forward_folds(_dates(), n_folds=4, initial_train_frac=0.5, val_frac=0.05, expanding=False)
    sizes = [len(f.train_idx) for f in folds]
    # rolling window should not grow anywhere near as much as expanding
    expanding_folds = generate_walk_forward_folds(_dates(), n_folds=4, initial_train_frac=0.5, val_frac=0.05, expanding=True)
    expanding_sizes = [len(f.train_idx) for f in expanding_folds]
    assert sizes[-1] < expanding_sizes[-1]


def test_raises_when_not_enough_data_for_requested_folds():
    # n=20, initial_train_frac=0.5 -> 10 rows remaining; n_folds=15 -> test_size = 10//15 = 0
    with pytest.raises(ValueError):
        generate_walk_forward_folds(_dates(n=20), n_folds=15, initial_train_frac=0.5, val_frac=0.05)


def test_last_fold_captures_remainder_of_series():
    dates = _dates(n=997)  # deliberately not evenly divisible
    folds = generate_walk_forward_folds(dates, n_folds=5, initial_train_frac=0.5, val_frac=0.05)
    assert folds[-1].test_idx.max() == dates.max()
