"""Integration test: the REAL end-to-end MLOps pipeline against real SPY data -- trains
a RAMA model, persists a checkpoint, computes a live prediction with XAI, and persists it
-- using an isolated temp SQLite DB (never the real dev DB).
"""
import tempfile
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")

_tmp_dir = tempfile.mkdtemp(prefix="emafrs_test_mlops_db_")
_TEST_DB_PATH = Path(_tmp_dir) / "test_mlops.db"

from emafrs.db.session import configure_database  # noqa: E402

configure_database(f"sqlite:///{_TEST_DB_PATH.as_posix()}")

from emafrs.db.models import ModelVersion, PredictionRecord  # noqa: E402
from emafrs.db.session import get_session, init_db  # noqa: E402
from emafrs.mlops.model_registry import load_latest_model, save_model  # noqa: E402
from emafrs.mlops.training_pipeline import run_training_and_prediction_pipeline  # noqa: E402
from emafrs.models.fusion import StaticFusionModel  # noqa: E402

init_db()

pytestmark = pytest.mark.integration

TICKER = "AAPL"


def test_model_registry_roundtrip():
    model = StaticFusionModel({"financial": 5}, hidden_dim=8, n_classes=3)
    version = save_model(
        model, "static", "ROUNDTRIP_TEST", {"financial": 5}, n_regimes=None,
        hidden_dim=8, n_classes=3, metrics={"accuracy": 0.42},
    )
    assert version.id is not None
    assert Path(version.checkpoint_path).exists()

    loaded_model, loaded_version = load_latest_model("static", "ROUNDTRIP_TEST")
    assert loaded_version.id == version.id
    assert loaded_version.metrics_json["accuracy"] == 0.42

    import torch
    x = torch.randn(1, 5)
    with torch.no_grad():
        original_logits, _ = model({"financial": x})
        loaded_logits, _ = loaded_model({"financial": x})
    assert torch.allclose(original_logits, loaded_logits)


def test_model_registry_deactivates_old_versions():
    model1 = StaticFusionModel({"financial": 5}, hidden_dim=8, n_classes=3)
    v1 = save_model(model1, "static", "DEACTIVATE_TEST", {"financial": 5}, None, 8, 3, {"accuracy": 0.3})
    model2 = StaticFusionModel({"financial": 5}, hidden_dim=8, n_classes=3)
    v2 = save_model(model2, "static", "DEACTIVATE_TEST", {"financial": 5}, None, 8, 3, {"accuracy": 0.4})

    session = get_session()
    try:
        refreshed_v1 = session.get(ModelVersion, v1.id)
        assert refreshed_v1.is_active is False
        refreshed_v2 = session.get(ModelVersion, v2.id)
        assert refreshed_v2.is_active is True
    finally:
        session.close()

    _, active_version = load_latest_model("static", "DEACTIVATE_TEST")
    assert active_version.id == v2.id


def test_full_training_and_prediction_pipeline_real_data():
    record = run_training_and_prediction_pipeline(TICKER, epochs=20, patience=5)

    assert record.id is not None
    assert record.ticker == TICKER
    assert record.direction in (-1, 0, 1)
    assert 0.0 <= record.confidence <= 1.0
    assert record.regime >= 1
    assert record.regime_name in ("Low", "Medium", "High", "Extreme")
    assert isinstance(record.modality_weights_json, dict) and len(record.modality_weights_json) > 0
    assert abs(sum(record.modality_weights_json.values()) - 1.0) < 1e-3
    assert isinstance(record.top_attributions_json, dict) and len(record.top_attributions_json) == 5
    # signal_consistency: a REAL computed value (in [-1, 1]) when >=2 modalities actually
    # contributed evidence to this real prediction, else honestly None (undefined for a
    # single modality, never a fabricated placeholder) -- checked against whichever case
    # this real run actually landed in, not assumed.
    if len(record.modality_weights_json) >= 2:
        assert record.signal_consistency is not None
        assert -1.0 <= record.signal_consistency <= 1.0
    else:
        assert record.signal_consistency is None
    assert record.explanation_text  # non-empty
    if record.counterfactual_json is not None:
        assert "causal claim" in record.explanation_text  # anti-overclaiming wording enforced by nlg.py
    assert record.model_version_id is not None

    # verify it was actually persisted, not just returned in-memory
    session = get_session()
    try:
        fetched = session.get(PredictionRecord, record.id)
        assert fetched is not None
        assert fetched.ticker == TICKER
    finally:
        session.close()

    print(f"\nReal pipeline explanation for {TICKER}:\n{record.explanation_text}")
