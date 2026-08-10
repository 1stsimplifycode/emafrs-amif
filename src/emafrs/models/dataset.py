"""Assembles a multimodal daily dataset for the RAMA/VAFE model and baselines.

Design decisions (each disclosed rather than silently made):

1. Regime HMM is fit ONLY on the training split, then applied (`.predict_proba`) forward
   across train+val+test. Fitting on the full series (including test dates) would leak
   future volatility-distribution information into regime labels used at test time --
   the single most common real-world leakage bug in regime-conditioned models.

2. Technical/risk features use ROLLING (not global) z-scoring (features/preprocessing.py),
   which is causal by construction and needs no separate train-only fit step.

3. Microstructure (MMIE) coverage is inherently thin: yfinance only exposes 1-minute bars
   for the trailing 7 days regardless of account tier (see docs/LIMITATIONS.md). Those
   ~5-7 trading days are resampled to daily and joined in; all other (older) rows get
   feature=0 and an explicit `microstructure_available` mask=0, rather than being
   imputed/faked. This is a genuine, disclosed coverage limitation, not a bug -- it will
   improve automatically as more live-pipeline history accumulates over time.

4. Text (ASEM) modality is NOT included in this dataset by default: as of this phase no
   NewsAPI/YouTube key is configured, so its availability mask would be 0 for every row,
   which is not a meaningful modality to train or ablate on yet. `include_text=True`
   requires the caller to supply pre-computed affective features (see `text_features`
   param) once real data exists -- this function never fabricates them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from emafrs.data.yfinance_client import YFinanceClient
from emafrs.features.preprocessing import rolling_zscore_frame
from emafrs.features.risk_metrics import rolling_cvar, rolling_historical_var, rolling_residual_volatility
from emafrs.features.technical import compute_all_technical_features, compute_returns
from emafrs.microstructure.proxies import build_microstructure_features
from emafrs.models.labels import build_ternary_labels, drop_undefined_labels
from emafrs.regime.hmm_regime import RegimeModel, build_regime_features, fit_regime_hmm


@dataclass
class DatasetBundle:
    features: pd.DataFrame          # all modality columns, chronologically indexed
    labels: pd.Series               # {-1, 0, 1}
    financial_columns: list[str]
    regime_columns: list[str]       # regime posterior probability columns
    microstructure_columns: list[str]
    train_idx: pd.DatetimeIndex
    val_idx: pd.DatetimeIndex
    test_idx: pd.DatetimeIndex
    regime_model: RegimeModel
    ticker: str
    horizon: int
    deadzone: float


def _chronological_split(index: pd.DatetimeIndex, train_frac: float, val_frac: float):
    n = len(index)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return index[:train_end], index[train_end:val_end], index[val_end:]


def build_daily_dataset(
    ticker: str,
    period: str = "5y",
    horizon: int = 1,
    deadzone: float = 0.0025,
    n_regime_states: int = 4,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    zscore_window: int = 126,
    include_microstructure: bool = True,
) -> DatasetBundle:
    """Fetches OHLCV (+ intraday, if requested) fresh and builds a dataset. For walk-
    forward validation across many folds of the SAME ticker, prefer
    `build_daily_dataset_from_ohlcv` directly with a pre-fetched (and, per fold, date-
    truncated) DataFrame to avoid re-fetching from yfinance on every fold."""
    df, _ = YFinanceClient().get_ohlcv(ticker, period=period, interval="1d")
    intraday_df = None
    if include_microstructure:
        try:
            intraday_df, _ = YFinanceClient().get_intraday_1m(ticker)
        except Exception:
            intraday_df = None
    return build_daily_dataset_from_ohlcv(
        df, ticker, horizon=horizon, deadzone=deadzone, n_regime_states=n_regime_states,
        train_frac=train_frac, val_frac=val_frac, zscore_window=zscore_window,
        include_microstructure=include_microstructure, intraday_df=intraday_df,
    )


def build_daily_dataset_from_ohlcv(
    df: pd.DataFrame,
    ticker: str,
    horizon: int = 1,
    deadzone: float = 0.0025,
    n_regime_states: int = 4,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    zscore_window: int = 126,
    include_microstructure: bool = True,
    intraday_df: pd.DataFrame | None = None,
    explicit_split: tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex] | None = None,
) -> DatasetBundle:
    """Same construction as build_daily_dataset, but takes an already-fetched daily OHLCV
    DataFrame (and optionally a pre-fetched intraday DataFrame) directly -- lets
    walk-forward validation truncate one fetched series to each fold's "as-of" date without
    a fresh network call per fold. `include_microstructure=True` with `intraday_df=None`
    will still attempt to fetch intraday data live for `ticker` (single-fold convenience
    path); pass `include_microstructure=False` for historical folds, since yfinance's
    7-day intraday cap means older folds could never have had real microstructure data
    "as of" their own test date anyway -- faking availability for them would be leakage
    disguised as a feature.

    `explicit_split`: pass (train_idx, val_idx, test_idx) from
    `emafrs.experiments.walk_forward.generate_walk_forward_folds` to use those exact
    boundaries (e.g. an EXPANDING training window) instead of this function's own
    train_frac/val_frac proportional split -- required for walk-forward validation, where
    the training window is not a fixed fraction of each fold's truncated data."""
    returns = compute_returns(df["close"])

    financial_raw = compute_all_technical_features(df)
    var95 = rolling_historical_var(returns, window=63)
    cvar95 = rolling_cvar(returns, window=63)
    financial_raw["var_95"] = var95
    financial_raw["cvar_95"] = cvar95

    financial = rolling_zscore_frame(financial_raw, window=zscore_window)
    financial_columns = list(financial.columns)

    regime_input = build_regime_features(returns, volume=df["volume"])
    dates_with_regime_features = regime_input.dropna().index
    if explicit_split is not None:
        train_dates_for_regime = dates_with_regime_features.intersection(explicit_split[0])
    else:
        train_dates_for_regime, _, _ = _chronological_split(
            dates_with_regime_features, train_frac, val_frac
        )
    regime_train_features = regime_input.loc[train_dates_for_regime]
    regime_model, _ = fit_regime_hmm(
        regime_train_features, vol_column="realized_vol", n_states=n_regime_states
    )
    regime_probs = regime_model.predict_proba(regime_input)
    regime_probs.columns = [f"regime_p{c}" for c in regime_probs.columns]
    regime_columns = list(regime_probs.columns)

    combined = financial.join(regime_probs, how="inner")

    microstructure_columns: list[str] = []
    if include_microstructure:
        try:
            intraday = intraday_df
            if intraday is None:
                intraday, _ = YFinanceClient().get_intraday_1m(ticker)
            micro = build_microstructure_features(intraday, vol_window=21)
            micro_daily = micro.resample("D").last().dropna(how="all")
            micro_daily.index = micro_daily.index.normalize()
            micro_daily_z = rolling_zscore_frame(micro_daily, window=min(zscore_window, max(3, len(micro_daily) - 1)))
            micro_daily_z = micro_daily_z.add_prefix("micro_")
            microstructure_columns = list(micro_daily_z.columns) + ["microstructure_available"]

            combined = combined.join(micro_daily_z, how="left")
            combined["microstructure_available"] = combined[micro_daily_z.columns].notna().any(axis=1).astype(float)
            combined[micro_daily_z.columns] = combined[micro_daily_z.columns].fillna(0.0)
        except Exception:
            # Microstructure is an enhancement, not a requirement -- if intraday data is
            # unavailable (e.g. delisted ticker, provider outage), proceed without it
            # rather than failing the whole dataset build, but do NOT fabricate values.
            combined["microstructure_available"] = 0.0
            microstructure_columns = ["microstructure_available"]

    labels = build_ternary_labels(df["close"], horizon=horizon, deadzone=deadzone)
    features_final, y = drop_undefined_labels(combined, labels)

    if explicit_split is not None:
        train_idx = features_final.index.intersection(explicit_split[0])
        val_idx = features_final.index.intersection(explicit_split[1])
        test_idx = features_final.index.intersection(explicit_split[2])
    else:
        train_idx, val_idx, test_idx = _chronological_split(features_final.index, train_frac, val_frac)

    return DatasetBundle(
        features=features_final,
        labels=y,
        financial_columns=financial_columns,
        regime_columns=regime_columns,
        microstructure_columns=microstructure_columns,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        regime_model=regime_model,
        ticker=ticker,
        horizon=horizon,
        deadzone=deadzone,
    )
