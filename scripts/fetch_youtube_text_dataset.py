"""Fetches REAL YouTube financial-communication videos + transcripts, runs them through
ASEM, and trains a 3-modality (financial + microstructure + text) RAMA model on the
result -- the first real test of the text modality this reconstruction has had.

Quota-conscious: a 6-month window in 30-day chunks = ~6 search.list calls = ~600 of the
10,000/day YouTube quota budget. Writes real results to
docs/_text_modality_results.md.
"""
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from emafrs.data.yfinance_client import YFinanceClient
from emafrs.models.baselines import encode_labels
from emafrs.models.dataset import build_daily_dataset_from_ohlcv
from emafrs.models.train import train_fusion_model
from emafrs.nlp.text_dataset import aggregate_daily_text_features, fetch_video_affective_records

ROOT = Path(__file__).resolve().parents[1]
TICKER = "SPY"
QUERY = "Federal Reserve interest rate"


def main():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=182)  # ~6 months

    print(f"Searching YouTube for {QUERY!r} from {start.date()} to {end.date()}...")
    records, stats = fetch_video_affective_records(
        QUERY, start, end, window_days=30, max_results_per_window=5, max_chars=2000
    )
    print(f"windows searched={stats.n_windows_searched} videos found={stats.n_videos_found} "
          f"transcripts available={stats.n_transcripts_available} processed={stats.n_videos_processed} "
          f"youtube_quota_units_used={stats.youtube_quota_units_used}")

    for r in records:
        print(f"  {r['published_at']}  sentiment={r['sentiment_polarity']:+.3f}  hype={r['hype_score']:.3f}  {r['title'][:70]}")

    text_daily = aggregate_daily_text_features(records)
    print(f"\nDaily text feature rows: {len(text_daily)}")
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

    print(f"\nModalities in this dataset: {list(modality_cols)}")
    print(f"text_available rows in train/val/test: "
          f"{bundle.features.loc[bundle.train_idx, 'text_available'].sum() if 'text_available' in bundle.features else 'n/a'} / "
          f"{bundle.features.loc[bundle.val_idx, 'text_available'].sum() if 'text_available' in bundle.features else 'n/a'} / "
          f"{bundle.features.loc[bundle.test_idx, 'text_available'].sum() if 'text_available' in bundle.features else 'n/a'}")

    print("\nTraining Model C (static fusion, all available modalities)...")
    _, result_static, weights_static = train_fusion_model(bundle, "static", modality_cols, epochs=150, patience=15, seed=42)

    print("Training Model D (RAMA, all available modalities)...")
    _, result_rama, weights_rama = train_fusion_model(bundle, "rama", modality_cols, epochs=150, patience=15, seed=42)

    print(f"\n{'Model':<20s} {'Accuracy':>10s} {'MacroF1':>10s} {'AUC':>10s}")
    for r in [result_static, result_rama]:
        auc_str = f"{r.auc_ovr:.3f}" if r.auc_ovr is not None else "n/a"
        print(f"{r.model_name:<20s} {r.accuracy:>10.3f} {r.macro_f1:>10.3f} {auc_str:>10s}")

    print(f"\nModel C mean weights: {dict(zip(modality_cols, weights_static.mean(axis=0).round(3)))}")
    print(f"Model D mean weights: {dict(zip(modality_cols, weights_rama.mean(axis=0).round(3)))}")

    lines = ["# Text modality: real YouTube-derived dataset results\n\n"]
    lines.append(f"Query: {QUERY!r}, ticker: {TICKER}, window: {start.date()} to {end.date()}.\n\n")
    lines.append(f"YouTube quota used: {stats.youtube_quota_units_used} units "
                 f"({stats.n_windows_searched} search.list calls).\n")
    lines.append(f"Videos found: {stats.n_videos_found}. With transcript: {stats.n_transcripts_available}. "
                 f"Successfully processed by ASEM: {stats.n_videos_processed}.\n")
    lines.append(f"Daily aggregated text rows: {len(text_daily)}.\n\n")
    lines.append(f"Modalities used: {list(modality_cols)}.\n\n")
    lines.append("| Model | Accuracy | Macro F1 | AUC |\n|---|---:|---:|---:|\n")
    for r in [result_static, result_rama]:
        auc_str = f"{r.auc_ovr:.3f}" if r.auc_ovr is not None else "n/a"
        lines.append(f"| {r.model_name} | {r.accuracy:.3f} | {r.macro_f1:.3f} | {auc_str} |\n")
    lines.append(f"\nModel C (static) mean fusion weights: {dict(zip(modality_cols, weights_static.mean(axis=0).round(3)))}\n")
    lines.append(f"Model D (RAMA) mean fusion weights: {dict(zip(modality_cols, weights_rama.mean(axis=0).round(3)))}\n")

    out_path = ROOT / "docs" / "_text_modality_results.md"
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
