"""Walk-forward validation fold generation -- build brief Section 26.

Generates a sequence of (train, val, test) chronological windows that advance through
time, rather than a single static split. Test windows across folds are strictly ordered
and non-overlapping; within each fold, train < val < test chronologically.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class WalkForwardFold:
    fold_id: int
    train_idx: pd.DatetimeIndex
    val_idx: pd.DatetimeIndex
    test_idx: pd.DatetimeIndex


def generate_walk_forward_folds(
    index: pd.DatetimeIndex,
    n_folds: int,
    initial_train_frac: float = 0.5,
    val_frac: float = 0.1,
    expanding: bool = True,
) -> list[WalkForwardFold]:
    """`expanding=True` (default): each fold's training window grows to include all data
    from the start up to that fold's validation window (standard expanding-window walk-
    forward). `expanding=False`: fixed-size rolling training window instead.

    The series (after the initial `initial_train_frac` warm-up) is divided into
    `n_folds` equal-sized test blocks; `val_frac` (of the FULL series length) immediately
    precedes each test block as that fold's validation window.
    """
    n = len(index)
    first_test_start = int(n * initial_train_frac)
    val_size = max(int(n * val_frac), 1)
    remaining = n - first_test_start
    test_size = remaining // n_folds
    if test_size < 1:
        raise ValueError(
            f"Not enough data for {n_folds} folds after the initial {initial_train_frac:.0%} "
            f"warm-up period ({remaining} rows remaining)."
        )

    folds = []
    for fold_id in range(n_folds):
        test_start = first_test_start + fold_id * test_size
        test_end = test_start + test_size if fold_id < n_folds - 1 else n  # last fold takes the remainder
        val_start = max(test_start - val_size, 0)
        train_start = 0 if expanding else max(val_start - int(n * initial_train_frac), 0)

        train_idx = index[train_start:val_start]
        val_idx = index[val_start:test_start]
        test_idx = index[test_start:test_end]

        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            continue

        folds.append(WalkForwardFold(fold_id=fold_id, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx))

    return folds
