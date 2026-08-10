"""Training loop for Model C (static fusion) and Model D (RAMA), shared over a
DatasetBundle (emafrs.models.dataset.build_daily_dataset)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from emafrs.models.baselines import EvalResult, encode_labels, evaluate_predictions
from emafrs.models.dataset import DatasetBundle
from emafrs.models.fusion import RAMAFusionModel, StaticFusionModel


def _modality_tensors(
    bundle: DatasetBundle, idx: pd.DatetimeIndex, modality_cols: dict[str, list[str]]
) -> dict[str, torch.Tensor]:
    return {
        name: torch.tensor(bundle.features.loc[idx, cols].to_numpy(), dtype=torch.float32)
        for name, cols in modality_cols.items()
    }


def train_fusion_model(
    bundle: DatasetBundle,
    model_type: str,
    modality_cols: dict[str, list[str]],
    hidden_dim: int = 32,
    epochs: int = 150,
    lr: float = 1e-3,
    patience: int = 12,
    seed: int = 42,
) -> tuple[torch.nn.Module, EvalResult, np.ndarray]:
    """model_type: 'static' (Model C) or 'rama' (Model D). Returns (trained model,
    test-set EvalResult, test-set per-row modality weights -- the observable fusion
    weights the build brief requires be exposed for every prediction)."""
    if model_type not in ("static", "rama"):
        raise ValueError(f"model_type must be 'static' or 'rama', got {model_type!r}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    modality_dims = {name: len(cols) for name, cols in modality_cols.items()}

    X_train = _modality_tensors(bundle, bundle.train_idx, modality_cols)
    X_val = _modality_tensors(bundle, bundle.val_idx, modality_cols)
    X_test = _modality_tensors(bundle, bundle.test_idx, modality_cols)

    y_train = torch.tensor(encode_labels(bundle.labels.loc[bundle.train_idx]), dtype=torch.long)
    y_val = torch.tensor(encode_labels(bundle.labels.loc[bundle.val_idx]), dtype=torch.long)
    y_test = torch.tensor(encode_labels(bundle.labels.loc[bundle.test_idx]), dtype=torch.long)

    regime_train = regime_val = regime_test = None
    if model_type == "rama":
        regime_train = torch.tensor(bundle.features.loc[bundle.train_idx, bundle.regime_columns].to_numpy(), dtype=torch.float32)
        regime_val = torch.tensor(bundle.features.loc[bundle.val_idx, bundle.regime_columns].to_numpy(), dtype=torch.float32)
        regime_test = torch.tensor(bundle.features.loc[bundle.test_idx, bundle.regime_columns].to_numpy(), dtype=torch.float32)
        model = RAMAFusionModel(modality_dims, n_regimes=len(bundle.regime_columns), hidden_dim=hidden_dim)
    else:
        model = StaticFusionModel(modality_dims, hidden_dim=hidden_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    def _forward(inputs, regime):
        if model_type == "rama":
            return model(inputs, regime)
        return model(inputs)

    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0

    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = _forward(X_train, regime_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = _forward(X_val, regime_val)
            val_loss = loss_fn(val_logits, y_val).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_logits, test_weights = _forward(X_test, regime_test)
        test_proba = torch.softmax(test_logits, dim=1).numpy()
        test_pred = test_proba.argmax(axis=1)

    result = evaluate_predictions(f"model_{model_type}", y_test.numpy(), test_pred, test_proba)
    return model, result, test_weights.numpy()
