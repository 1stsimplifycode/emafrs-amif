"""Covers the optimizer construction detail in train_fusion_model: the fusion-gate
parameter (RAMAFusionModel.gate / StaticFusionModel.static_logits) must be excluded from
weight decay -- see docs/EXPERIMENT_PROTOCOL.md's fusion-gate diagnostic for why this was
added (tested as a fix for the flat-gate finding; ruled out as the cause, but kept as a
defensible practice on its own merits)."""
import numpy as np
import pandas as pd

from emafrs.models.dataset import build_daily_dataset_from_ohlcv
from emafrs.models.train import train_fusion_model


def _synthetic_ohlcv(n: int = 300, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates)


def test_rama_gate_and_static_logits_excluded_from_weight_decay(monkeypatch):
    captured = {}
    import torch

    real_adam_init = torch.optim.Adam.__init__

    def spy_init(self, params, *args, **kwargs):
        param_groups = list(params)
        captured["groups"] = param_groups
        return real_adam_init(self, param_groups, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "__init__", spy_init)

    df = _synthetic_ohlcv()
    bundle = build_daily_dataset_from_ohlcv(df, "SYN", include_microstructure=False)
    modality_cols = {"financial": bundle.financial_columns}

    train_fusion_model(bundle, "rama", modality_cols, epochs=2, patience=1, seed=0)

    groups = captured["groups"]
    assert len(groups) == 2
    zero_decay_group = next(g for g in groups if g["weight_decay"] == 0.0)
    decayed_group = next(g for g in groups if g["weight_decay"] > 0.0)
    assert len(zero_decay_group["params"]) == 1  # just RAMAFusionModel.gate
    assert len(decayed_group["params"]) > 1
