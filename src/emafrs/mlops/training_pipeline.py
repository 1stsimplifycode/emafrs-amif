"""End-to-end training + live prediction pipeline: build dataset -> train RAMA -> persist
checkpoint -> compute XAI for the most recent available row -> persist the prediction.

This is the function the API's predict endpoint calls (synchronously, for a demo-scale
deployment) and that emafrs.mlops.scheduler calls periodically for a production-like
refresh pattern. Both paths go through this one function so there is a single source of
truth for "how a prediction gets made."
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import torch

from emafrs.db.models import PredictionRecord
from emafrs.db.session import get_session
from emafrs.mlops.model_registry import save_model
from emafrs.models.dataset import build_daily_dataset_from_ohlcv
from emafrs.models.train import train_fusion_model
from emafrs.regime.hmm_regime import REGIME_NAMES
from emafrs.risk.cross_modal_conflict import compute_agreement, modality_direction_from_attributions
from emafrs.xai.counterfactual import search_all_feature_counterfactuals
from emafrs.xai.integrated_gradients import compute_integrated_gradients
from emafrs.xai.nlg import generate_explanation

logger = logging.getLogger("emafrs.mlops")

DIRECTION_FROM_CLASS_IDX = {0: -1, 1: 0, 2: 1}


def _fetch_real_text_daily(ticker: str, trading_days):
    """Real, opt-in text-modality fetch for a live prediction: channel-targeted YouTube
    (CNBC/Bloomberg, live-verified channel IDs) + domain-targeted NewsAPI (cnbc.com,
    bloomberg.com), over NewsAPI's real ~29-day lookback window (querying further back
    is pointless on the free tier -- see emafrs.nlp.text_dataset's module docstring).
    Mapped forward-only (never backward) onto `trading_days` so non-trading-day articles
    aren't silently dropped (see aggregate_daily_text_features's `trading_days` param).

    Every real pilot run to date (docs/TEXT_MODALITY_PILOT.md) has still ended up with
    text_available=0 in every split -- either from YouTube's IP-blocking or NewsAPI's
    z-score-warm-up-vs-lookback-window mismatch. This function does not fix that; it
    makes the attempt real and repeatable (and channel/domain-targeted, unlike the
    original pilots' generic keyword search) rather than skipping the attempt entirely.
    Returns whatever real data resulted, honestly -- often an empty-but-correctly-shaped
    frame, never a fabricated one."""
    from emafrs.data.newsapi_client import KNOWN_FINANCIAL_NEWS_DOMAINS
    from emafrs.nlp.asem import AffectiveSignalExtractionModule
    from emafrs.nlp.text_dataset import (
        aggregate_daily_text_features,
        fetch_channel_video_affective_records,
        fetch_news_affective_records,
    )

    now = datetime.now(timezone.utc)
    asem = AffectiveSignalExtractionModule()

    yt_records, yt_stats = fetch_channel_video_affective_records(
        now - timedelta(days=7), now, max_results_per_channel=5, max_chars=2000, asem=asem,
    )
    logger.info(
        "Live text fetch (YouTube, channel-targeted): found=%d processed=%d quota=%d",
        yt_stats.n_videos_found, yt_stats.n_videos_processed, yt_stats.youtube_quota_units_used,
    )

    news_records, news_stats = fetch_news_affective_records(
        ticker, now - timedelta(days=29), now, page_size=20, max_chars=1000,
        asem=asem, domains=KNOWN_FINANCIAL_NEWS_DOMAINS,
    )
    logger.info(
        "Live text fetch (NewsAPI, domain-targeted): found=%d processed=%d",
        news_stats.n_articles_found, news_stats.n_articles_processed,
    )

    combined = yt_records + news_records
    return aggregate_daily_text_features(combined, trading_days=trading_days)


def run_training_and_prediction_pipeline(
    ticker: str, epochs: int = 100, patience: int = 12, seed: int = 42, include_text: bool = False,
) -> PredictionRecord:
    """Trains a fresh RAMA model on `ticker` and produces a persisted prediction (with
    explanation) for the most recent row in the dataset. Real computation, real data --
    nothing here is templated or precomputed.

    `include_text`: opt-in (default False, preserving the existing fast/quota-free
    behavior). When True, attempts a real, channel/domain-targeted text fetch (see
    `_fetch_real_text_daily`) before building the dataset, so "text" becomes a genuine
    candidate modality if -- and only if -- real data actually supports it for this
    specific run. Adds real network latency and consumes real YouTube/NewsAPI quota."""
    from emafrs.data.yfinance_client import YFinanceClient

    logger.info("Fetching OHLCV for %s", ticker)
    df, _ = YFinanceClient().get_ohlcv(ticker, period="5y", interval="1d")
    intraday_df = None
    try:
        intraday_df, _ = YFinanceClient().get_intraday_1m(ticker)
    except Exception:
        intraday_df = None

    text_daily = None
    if include_text:
        try:
            text_daily = _fetch_real_text_daily(ticker, df.index)
        except Exception as exc:
            logger.warning("Live text fetch failed for %s, proceeding without it: %s", ticker, exc)
            text_daily = None

    logger.info("Building dataset for %s", ticker)
    bundle = build_daily_dataset_from_ohlcv(
        df, ticker, horizon=1, deadzone=0.0025, include_microstructure=True,
        intraday_df=intraday_df, text_daily=text_daily,
    )

    modality_cols = {"financial": bundle.financial_columns}
    if len(bundle.microstructure_columns) > 1:
        modality_cols["microstructure"] = bundle.microstructure_columns
    if len(bundle.text_columns) > 1:  # more than just the always-present "text_available" flag column
        modality_cols["text"] = bundle.text_columns

    logger.info("Training RAMA model for %s", ticker)
    model, result, _weights = train_fusion_model(
        bundle, "rama", modality_cols, epochs=epochs, patience=patience, seed=seed
    )

    modality_dims = {name: len(cols) for name, cols in modality_cols.items()}
    version = save_model(
        model, "rama", ticker, modality_dims, n_regimes=len(bundle.regime_columns),
        hidden_dim=32, n_classes=3,
        metrics={"accuracy": result.accuracy, "macro_f1": result.macro_f1, "auc_ovr": result.auc_ovr, "n_test": result.n_test},
    )
    logger.info("Saved model version %s (accuracy=%.3f)", version.id, result.accuracy)

    # Predict on the MOST RECENT available row (the closest thing to "today" this dataset supports)
    latest_date = bundle.features.index.max()
    modality_inputs = {
        name: torch.tensor(bundle.features.loc[[latest_date], cols].to_numpy(), dtype=torch.float32)
        for name, cols in modality_cols.items()
    }
    regime_probs = torch.tensor(
        bundle.features.loc[[latest_date], bundle.regime_columns].to_numpy(), dtype=torch.float32
    )

    model.eval()
    with torch.no_grad():
        logits, weights = model(modality_inputs, regime_probs)
        proba = torch.softmax(logits, dim=1)[0]
        pred_class = int(proba.argmax())

    attrs = compute_integrated_gradients(
        model, modality_inputs, list(modality_cols), target_class=pred_class,
        regime_probs=regime_probs, n_steps=30,
    )
    financial_attrs = list(zip(bundle.financial_columns, attrs["financial"][0]))
    top_financial = sorted(financial_attrs, key=lambda kv: -abs(kv[1]))[:5]

    # REAL cross-modal signal consistency, computed for every live prediction where more
    # than one modality actually contributed evidence -- derived from the same IG
    # attributions already computed above (nothing new invented), fed into the existing,
    # separately-tested compute_agreement(). Left as None (never a fabricated number)
    # when only one modality is present, since "agreement" between one thing and nothing
    # is undefined, not "unavailable due to missing computation."
    modality_attrs = {name: attrs[name] for name in modality_cols}
    modality_directions = modality_direction_from_attributions(modality_attrs)
    signal_consistency = compute_agreement(modality_directions).mean_agreement if len(modality_directions) >= 2 else None

    fixed_extra = {k: v for k, v in modality_inputs.items() if k != "financial"}

    def predict_fn(financial_vec):
        with torch.no_grad():
            inp = dict(fixed_extra)
            inp["financial"] = torch.tensor(financial_vec, dtype=torch.float32).unsqueeze(0)
            logit, _ = model(inp, regime_probs)
            return int(torch.softmax(logit, dim=1).argmax(dim=1).item())

    x0 = modality_inputs["financial"][0].numpy()
    cf_results = search_all_feature_counterfactuals(
        predict_fn, x0, bundle.financial_columns, search_range=(-6.0, 6.0), n_steps=25
    )
    top_cf = cf_results[0] if cf_results else None

    regime_idx = int(regime_probs.argmax(dim=1).item()) + 1
    regime_name = REGIME_NAMES.get(regime_idx, str(regime_idx))
    modality_weight_dict = dict(zip(modality_cols.keys(), weights[0].tolist()))

    explanation = generate_explanation(
        direction=DIRECTION_FROM_CLASS_IDX[pred_class],
        confidence=float(proba[pred_class]),
        regime_name=regime_name,
        modality_weights=modality_weight_dict,
        top_financial_attributions=top_financial,
        counterfactual=top_cf,
    )

    session = get_session()
    try:
        record = PredictionRecord(
            ticker=ticker,
            as_of_date=str(latest_date.date()),
            direction=DIRECTION_FROM_CLASS_IDX[pred_class],
            confidence=float(proba[pred_class]),
            regime=regime_idx,
            regime_name=regime_name,
            modality_weights_json=modality_weight_dict,
            top_attributions_json={name: float(val) for name, val in top_financial},
            counterfactual_json=(
                {
                    "feature_name": top_cf.feature_name, "delta": top_cf.delta,
                    "original_value": top_cf.original_value, "counterfactual_value": top_cf.counterfactual_value,
                }
                if top_cf is not None else None
            ),
            manipulation_risk=None,  # computed separately once real hype/microstructure signals are wired end-to-end live
            signal_consistency=signal_consistency,
            explanation_text=explanation,
            model_version_id=version.id,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record
    finally:
        session.close()
