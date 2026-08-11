"""FastAPI endpoint tests against an isolated temp SQLite DB (never the real dev DB at
data/emafrs.db). DB isolation must happen BEFORE importing emafrs.api.app, since
emafrs.db.session binds its engine at import time.
"""
import tempfile
from pathlib import Path

import pytest

_tmp_dir = tempfile.mkdtemp(prefix="emafrs_test_db_")
_TEST_DB_PATH = Path(_tmp_dir) / "test_emafrs.db"

from emafrs.db.session import configure_database  # noqa: E402

configure_database(f"sqlite:///{_TEST_DB_PATH.as_posix()}")

from fastapi.testclient import TestClient  # noqa: E402

from emafrs.api.app import app  # noqa: E402
from emafrs.db.models import ModelVersion, PredictionRecord  # noqa: E402
from emafrs.db.session import get_session, init_db  # noqa: E402

init_db()

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clean_db():
    """Each test starts with an empty predictions/model_versions table."""
    session = get_session()
    try:
        session.query(PredictionRecord).delete()
        session.query(ModelVersion).delete()
        session.commit()
    finally:
        session.close()
    yield


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_data_sources_health_default_does_not_hit_network():
    """live defaults to False -- must not attempt real network calls, only report
    configuration status, so this test must be fast and not flaky on network issues."""
    resp = client.get("/api/v1/data-sources/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["checked_live"] is False
    provider_names = {s["provider"] for s in body["sources"]}
    assert {"yfinance", "sec_edgar", "newsapi", "youtube", "alphavantage", "finnhub", "fred"}.issubset(provider_names)
    yfinance_entry = next(s for s in body["sources"] if s["provider"] == "yfinance")
    assert yfinance_entry["configured"] is True  # no key needed


def test_get_latest_prediction_404_when_none_exists():
    resp = client.get("/api/v1/predictions/ZZZZ_NOPRED/latest")
    assert resp.status_code == 404
    assert "No prediction has been computed yet" in resp.json()["detail"]


def test_prediction_history_empty_list_when_none_exists():
    resp = client.get("/api/v1/predictions/ZZZZ_NOPRED/history")
    assert resp.status_code == 200
    assert resp.json() == []


def test_model_versions_empty_list_when_none_exists():
    resp = client.get("/api/v1/model-versions/ZZZZ_NOPRED")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_latest_prediction_returns_persisted_record():
    """Directly inserts a prediction record (bypassing the real training pipeline, which
    is exercised separately in tests/integration/test_mlops_pipeline.py) to test the
    read-path/serialization contract in isolation."""
    session = get_session()
    try:
        record = PredictionRecord(
            ticker="TESTTICK", as_of_date="2026-01-01", direction=1, confidence=0.65,
            regime=2, regime_name="Medium",
            modality_weights_json={"financial": 0.6, "microstructure": 0.4},
            top_attributions_json={"rsi_14": 0.12},
            counterfactual_json={"feature_name": "rsi_14", "delta": 0.5, "original_value": 0.1, "counterfactual_value": 0.6},
            manipulation_risk=0.2, explanation_text="Predicted market direction: bullish.",
            model_version_id=None,
        )
        session.add(record)
        session.commit()
    finally:
        session.close()

    resp = client.get("/api/v1/predictions/testtick/latest")  # lowercase input must be normalized
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "TESTTICK"
    assert body["direction_label"] == "bullish"
    assert body["modality_weights"] == {"financial": 0.6, "microstructure": 0.4}
    assert body["counterfactual"]["feature_name"] == "rsi_14"
    assert body["signal_consistency"] is None  # not set on this record -- never defaulted to a fabricated number


def test_get_latest_prediction_surfaces_real_signal_consistency_when_computed():
    session = get_session()
    try:
        record = PredictionRecord(
            ticker="CONSISTTEST", as_of_date="2026-01-01", direction=1, confidence=0.7,
            regime=1, regime_name="Low",
            modality_weights_json={"financial": 0.5, "microstructure": 0.5},
            top_attributions_json={"rsi_14": 0.1},
            counterfactual_json=None,
            manipulation_risk=None, signal_consistency=0.42,
            explanation_text="Predicted market direction: bullish.",
            model_version_id=None,
        )
        session.add(record)
        session.commit()
    finally:
        session.close()

    resp = client.get("/api/v1/predictions/CONSISTTEST/latest")
    assert resp.status_code == 200
    assert resp.json()["signal_consistency"] == 0.42


def test_rate_limit_headers_do_not_break_normal_traffic():
    for _ in range(5):
        resp = client.get("/health")
        assert resp.status_code == 200


def test_security_headers_present():
    resp = client.get("/health")
    assert resp.headers.get("x-content-type-options") == "nosniff"
    assert resp.headers.get("x-frame-options") == "DENY"
