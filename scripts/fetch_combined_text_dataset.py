"""Fetches REAL text data from BOTH YouTube and NewsAPI, combines them into one daily
text-feature dataset, and trains a 3-modality (financial + microstructure + text) RAMA
model on the result -- the follow-up to scripts/fetch_youtube_text_dataset.py now that
NewsAPI is verified working.

NewsAPI's free tier only covers roughly the trailing month regardless of the date range
requested (see emafrs.nlp.text_dataset module docstring) -- a fundamentally different
coverage shape than YouTube's sparse-but-potentially-historical (and, in this
environment, IP-blocked) coverage. This script deliberately keeps the YouTube window
modest (2 months) since the prior 6-month/35-video run showed most requests get blocked
anyway; NewsAPI gets 2 queries over its own allowed ~29-day window.
"""
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from emafrs.data.yfinance_client import YFinanceClient
from emafrs.models.dataset import build_daily_dataset_from_ohlcv
from emafrs.models.train import train_fusion_model
from emafrs.nlp.asem import AffectiveSignalExtractionModule
from emafrs.nlp.text_dataset import (
    aggregate_daily_text_features,
    fetch_news_affective_records,
    fetch_video_affective_records,
)

ROOT = Path(__file__).resolve().parents[1]
TICKER = "SPY"
YOUTUBE_QUERY = "Federal Reserve interest rate"
NEWS_QUERIES = ["Federal Reserve interest rate", "S&P 500 stock market"]


def main():
    asem = AffectiveSignalExtractionModule()  # load once, share across both fetchers
    now = datetime.now(timezone.utc)

    print(f"=== YouTube: searching {YOUTUBE_QUERY!r} over the last 2 months ===")
    yt_records, yt_stats = fetch_video_affective_records(
        YOUTUBE_QUERY, now - timedelta(days=60), now, window_days=30,
        max_results_per_window=5, max_chars=2000, asem=asem,
    )
    print(f"YouTube: windows={yt_stats.n_windows_searched} found={yt_stats.n_videos_found} "
          f"transcripts={yt_stats.n_transcripts_available} processed={yt_stats.n_videos_processed} "
          f"quota_units={yt_stats.youtube_quota_units_used}")

    news_records = []
    for query in NEWS_QUERIES:
        print(f"\n=== NewsAPI: searching {query!r} over the last 29 days ===")
        recs, stats = fetch_news_affective_records(
            query, now - timedelta(days=29), now, page_size=50, max_chars=1000, asem=asem,
        )
        print(f"NewsAPI ({query!r}): found={stats.n_articles_found} processed={stats.n_articles_processed}")
        news_records.extend(recs)

    combined_records = yt_records + news_records
    print(f"\nCombined: {len(yt_records)} YouTube + {len(news_records)} NewsAPI = {len(combined_records)} total items")

    text_daily = aggregate_daily_text_features(combined_records)
    print(f"Daily aggregated text rows: {len(text_daily)}")
    print(text_daily)

    print(f"\nBuilding full dataset for {TICKER} (financial + microstructure + text)...")
    df, _ = YFinanceClient().get_ohlcv(TICKER, period="5y", interval="1d")
    intraday_df = None
    try:
        intraday_df, _ = YFinanceClient().get_intraday_1m(TICKER)
    except Exception:
        pass

    bundle = build_daily_dataset_from_ohlcv(
        df, TICKER, horizon=1, deadzone=0.0025, include_microstructure=True,
        intraday_df=intraday_df, text_daily=text_daily,
    )

    modality_cols = {"financial": bundle.financial_columns}
    if len(bundle.microstructure_columns) > 1:
        modality_cols["microstructure"] = bundle.microstructure_columns
    if len(bundle.text_columns) > 1:
        modality_cols["text"] = bundle.text_columns

    print(f"\nModalities: {list(modality_cols)}")
    for split_name, idx in [("train", bundle.train_idx), ("val", bundle.val_idx), ("test", bundle.test_idx)]:
        avail = bundle.features.loc[idx, "text_available"].sum() if "text_available" in bundle.features else "n/a"
        print(f"  text_available rows in {split_name}: {avail} / {len(idx)}")

    if "text" not in modality_cols:
        print("\nNo usable text modality (text_available never True after z-score warm-up) -- stopping here.")
        return

    print("\nTraining Model C (static fusion)...")
    _, result_static, weights_static = train_fusion_model(bundle, "static", modality_cols, epochs=150, patience=15, seed=42)
    print("Training Model D (RAMA)...")
    _, result_rama, weights_rama = train_fusion_model(bundle, "rama", modality_cols, epochs=150, patience=15, seed=42)

    print(f"\n{'Model':<20s} {'Accuracy':>10s} {'MacroF1':>10s} {'AUC':>10s}")
    for r in [result_static, result_rama]:
        auc_str = f"{r.auc_ovr:.3f}" if r.auc_ovr is not None else "n/a"
        print(f"{r.model_name:<20s} {r.accuracy:>10.3f} {r.macro_f1:>10.3f} {auc_str:>10s}")
    print(f"Static mean weights: {dict(zip(modality_cols, weights_static.mean(axis=0).round(3)))}")
    print(f"RAMA mean weights:   {dict(zip(modality_cols, weights_rama.mean(axis=0).round(3)))}")

    lines = ["# Combined YouTube + NewsAPI text modality results (real run)\n\n"]
    lines.append(f"YouTube query: {YOUTUBE_QUERY!r} (60d window). NewsAPI queries: {NEWS_QUERIES} (29d window). Ticker: {TICKER}.\n\n")
    lines.append(f"YouTube: found={yt_stats.n_videos_found} processed={yt_stats.n_videos_processed} quota={yt_stats.youtube_quota_units_used}.\n")
    lines.append(f"NewsAPI: {len(news_records)} articles processed across {len(NEWS_QUERIES)} queries.\n")
    lines.append(f"Combined daily text rows: {len(text_daily)}.\n\n")
    for split_name, idx in [("train", bundle.train_idx), ("val", bundle.val_idx), ("test", bundle.test_idx)]:
        avail = bundle.features.loc[idx, "text_available"].sum() if "text_available" in bundle.features else "n/a"
        lines.append(f"text_available rows in {split_name}: {avail} / {len(idx)}\n")
    lines.append(f"\nModalities used: {list(modality_cols)}.\n\n")
    lines.append("| Model | Accuracy | Macro F1 | AUC |\n|---|---:|---:|---:|\n")
    for r in [result_static, result_rama]:
        auc_str = f"{r.auc_ovr:.3f}" if r.auc_ovr is not None else "n/a"
        lines.append(f"| {r.model_name} | {r.accuracy:.3f} | {r.macro_f1:.3f} | {auc_str} |\n")
    lines.append(f"\nStatic mean weights: {dict(zip(modality_cols, weights_static.mean(axis=0).round(3)))}\n")
    lines.append(f"RAMA mean weights: {dict(zip(modality_cols, weights_rama.mean(axis=0).round(3)))}\n")

    out_path = ROOT / "docs" / "_combined_text_modality_results.md"
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
