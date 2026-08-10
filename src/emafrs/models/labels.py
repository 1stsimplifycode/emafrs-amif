"""Ternary market-direction label construction -- PDF Sec 6.7.3: y_t in {-1, 0, +1}.

LEAKAGE NOTE: y_t is necessarily a function of FUTURE prices (close_{t+h}), which is
correct and required for supervised learning, but it means:
  1. y_t must never be used as a FEATURE at time t (only as the training target).
  2. The trailing h rows of any dataset have an undefined label (no future price exists
     yet) and must be dropped, never filled/imputed.
Both are enforced in `build_labels` below and re-checked by
`emafrs.features.leakage_audit.check_chronological_split` at the train/test split stage.
"""
from __future__ import annotations

import pandas as pd


def forward_return(close: pd.Series, horizon: int = 1) -> pd.Series:
    """r_{t->t+h} = close_{t+h} / close_t - 1. The trailing `horizon` rows are NaN (no
    future price available) -- callers must drop them, not fill them."""
    return close.shift(-horizon) / close - 1.0


def build_ternary_labels(
    close: pd.Series,
    horizon: int = 1,
    deadzone: float = 0.0025,
) -> pd.Series:
    """y_t = +1 (bullish) if forward_return > deadzone; -1 (bearish) if < -deadzone; else 0
    (neutral). `deadzone` is a disclosed, fixed hyperparameter (default 0.25%, a modest
    fraction of typical large-cap daily volatility) -- NOT claimed to be optimal or
    empirically tuned; documented here rather than silently hardcoded elsewhere.

    Returns a Series of {-1, 0, 1} with NaN for the trailing `horizon` rows where no
    forward price exists yet.
    """
    fwd = forward_return(close, horizon=horizon)
    labels = pd.Series(0, index=close.index, dtype="float64")
    labels[fwd > deadzone] = 1
    labels[fwd < -deadzone] = -1
    labels[fwd.isna()] = float("nan")
    return labels


def drop_undefined_labels(features: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Drops rows where the label is undefined (trailing horizon) or any feature is NaN
    (warm-up period) -- the standard "usable rows" intersection for supervised training."""
    aligned = features.join(labels.rename("__label__"), how="inner")
    aligned = aligned.dropna()
    y = aligned.pop("__label__")
    return aligned, y
