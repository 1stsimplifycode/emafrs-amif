"""Model checkpoint persistence -- build brief Section 35: model versioning.

Saves a trained fusion model's state_dict + architecture metadata to
models/checkpoints/ (gitignored, regenerable by retraining) and records a queryable
ModelVersion row in the DB so old versions remain inspectable even after a newer one
becomes active.
"""
from __future__ import annotations

from pathlib import Path

import torch

from emafrs.db.models import ModelVersion
from emafrs.db.session import get_session
from emafrs.models.fusion import RAMAFusionModel, StaticFusionModel

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = _PROJECT_ROOT / "models" / "checkpoints"


def save_model(
    model: torch.nn.Module,
    model_type: str,
    ticker: str,
    modality_dims: dict[str, int],
    n_regimes: int | None,
    hidden_dim: int,
    n_classes: int,
    metrics: dict,
) -> ModelVersion:
    """Persists the model and deactivates any prior active version for the same
    (model_type, ticker) pair -- old checkpoints are kept on disk and in the DB (not
    deleted), just no longer served as "the" active model, per build brief Section 35's
    instruction not to automatically replace production models without safeguards."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    session = get_session()
    try:
        version = ModelVersion(
            model_type=model_type, ticker=ticker, checkpoint_path="", metrics_json=metrics,
            modality_columns_json={
                "modality_dims": modality_dims, "n_regimes": n_regimes,
                "hidden_dim": hidden_dim, "n_classes": n_classes,
            },
        )
        session.add(version)
        session.flush()  # assigns version.id without committing

        checkpoint_path = CHECKPOINT_DIR / f"{model_type}_{ticker}_{version.id}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        version.checkpoint_path = str(checkpoint_path)

        session.query(ModelVersion).filter(
            ModelVersion.model_type == model_type,
            ModelVersion.ticker == ticker,
            ModelVersion.id != version.id,
        ).update({"is_active": False})

        session.commit()
        session.refresh(version)
        return version
    finally:
        session.close()


def load_latest_model(model_type: str, ticker: str) -> tuple[torch.nn.Module | None, ModelVersion | None]:
    session = get_session()
    try:
        version = (
            session.query(ModelVersion)
            .filter(
                ModelVersion.model_type == model_type,
                ModelVersion.ticker == ticker,
                ModelVersion.is_active == True,  # noqa: E712
            )
            .order_by(ModelVersion.trained_at.desc())
            .first()
        )
        if version is None:
            return None, None

        meta = version.modality_columns_json
        modality_dims = meta["modality_dims"]
        if model_type == "rama":
            model = RAMAFusionModel(
                modality_dims, n_regimes=meta["n_regimes"], hidden_dim=meta["hidden_dim"], n_classes=meta["n_classes"]
            )
        else:
            model = StaticFusionModel(modality_dims, hidden_dim=meta["hidden_dim"], n_classes=meta["n_classes"])

        state_dict = torch.load(version.checkpoint_path, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model, version
    finally:
        session.close()
