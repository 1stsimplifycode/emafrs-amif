"""Real walk-forward validation across multiple folds and seeds, financial-only (the one
modality with enough history to actually walk forward -- see dataset.py's docstring on why
microstructure/text can't be included in a genuine multi-fold study yet).

Compares Model A (opaque XGBoost), Model B (explainable logistic regression), and a
single-modality neural fusion baseline, across 5 expanding-window folds x 3 seeds each,
with paired statistical significance testing. Writes results to
docs/_walk_forward_results.md (merged by hand into docs/EXPERIMENT_PROTOCOL.md).
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import torch

from emafrs.data.yfinance_client import YFinanceClient
from emafrs.experiments.significance import compare_paired_results, confidence_interval
from emafrs.experiments.walk_forward import generate_walk_forward_folds
from emafrs.models.baselines import ModelAOpaqueXGBoost, ModelBExplainableLogReg, encode_labels, evaluate_predictions
from emafrs.models.dataset import build_daily_dataset_from_ohlcv
from emafrs.models.train import train_fusion_model

ROOT = Path(__file__).resolve().parents[1]
TICKER = "SPY"
N_FOLDS = 5
SEEDS = [0, 1, 2]


def main():
    print(f"Fetching {TICKER} full history...")
    df, _ = YFinanceClient().get_ohlcv(TICKER, period="10y", interval="1d")
    print(f"n_rows={len(df)}  range={df.index.min()} -> {df.index.max()}")

    folds = generate_walk_forward_folds(df.index, n_folds=N_FOLDS, initial_train_frac=0.5, val_frac=0.08, expanding=True)
    print(f"Generated {len(folds)} walk-forward folds (expanding window).")

    results = {"A_xgboost": [], "B_logreg": [], "NN_single_modality": []}

    for fold in folds:
        print(f"\n--- Fold {fold.fold_id}: train n={len(fold.train_idx)} val n={len(fold.val_idx)} test n={len(fold.test_idx)} "
              f"test range {fold.test_idx.min().date()} -> {fold.test_idx.max().date()} ---")

        bundle = build_daily_dataset_from_ohlcv(
            df, TICKER, include_microstructure=False,
            explicit_split=(fold.train_idx, fold.val_idx, fold.test_idx),
        )
        if len(bundle.train_idx) < 50 or len(bundle.test_idx) < 10:
            print("  skipping fold: insufficient rows after feature warm-up")
            continue

        X_train = bundle.features.loc[bundle.train_idx, bundle.financial_columns]
        X_test = bundle.features.loc[bundle.test_idx, bundle.financial_columns]
        y_train = bundle.labels.loc[bundle.train_idx]
        y_test = bundle.labels.loc[bundle.test_idx]
        y_test_idx = encode_labels(y_test)

        model_a = ModelAOpaqueXGBoost(n_estimators=150).fit(X_train, y_train)
        proba_a = model_a.predict_proba(X_test)
        res_a = evaluate_predictions("A", y_test_idx, proba_a.argmax(axis=1), proba_a)
        results["A_xgboost"].append(res_a.accuracy)
        print(f"  Model A (XGBoost):  acc={res_a.accuracy:.3f} f1={res_a.macro_f1:.3f}")

        model_b = ModelBExplainableLogReg().fit(X_train, y_train)
        proba_b = model_b.predict_proba(X_test)
        res_b = evaluate_predictions("B", y_test_idx, proba_b.argmax(axis=1), proba_b)
        results["B_logreg"].append(res_b.accuracy)
        print(f"  Model B (LogReg):   acc={res_b.accuracy:.3f} f1={res_b.macro_f1:.3f}")

        seed_accs = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            _, res_nn, _ = train_fusion_model(
                bundle, "static", {"financial": bundle.financial_columns}, epochs=80, patience=10, seed=seed
            )
            seed_accs.append(res_nn.accuracy)
        mean_nn_acc = float(np.mean(seed_accs))
        results["NN_single_modality"].append(mean_nn_acc)
        print(f"  NN (single-modality, {len(SEEDS)} seeds): mean_acc={mean_nn_acc:.3f} (per-seed: {[round(a,3) for a in seed_accs]})")

    print("\n\n=== Walk-forward summary (mean accuracy +/- 95% CI across folds) ===")
    for name, values in results.items():
        if not values:
            continue
        mean, lo, hi = confidence_interval(values)
        print(f"{name:<22s} n_folds={len(values)}  mean_acc={mean:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]  per-fold={[round(v,3) for v in values]}")

    print("\n=== Paired significance tests (across folds) ===")
    pairs = [("A_xgboost", "B_logreg"), ("A_xgboost", "NN_single_modality"), ("B_logreg", "NN_single_modality")]
    for name_a, name_b in pairs:
        if len(results[name_a]) != len(results[name_b]) or len(results[name_a]) < 2:
            continue
        sig = compare_paired_results(results[name_a], results[name_b], metric_name="accuracy")
        print(f"{name_a} vs {name_b}: mean_diff={sig.mean_diff:+.3f}  wilcoxon_p={sig.wilcoxon_p}  "
              f"paired_t_p={sig.paired_t_p}  significant_at_0.05={sig.significant_at_05}")

    # Write markdown summary
    lines = ["# Walk-forward validation results (real run)\n\n"]
    lines.append(f"Ticker: {TICKER}. {len(folds)} expanding-window folds, {len(SEEDS)} seeds for the NN model.\n\n")
    lines.append("| Model | n_folds | Mean Accuracy | 95% CI | Per-fold accuracy |\n|---|---:|---:|---|---|\n")
    for name, values in results.items():
        if not values:
            continue
        mean, lo, hi = confidence_interval(values)
        lines.append(f"| {name} | {len(values)} | {mean:.3f} | [{lo:.3f}, {hi:.3f}] | {[round(v,3) for v in values]} |\n")
    lines.append("\n## Paired significance (Wilcoxon / paired t-test)\n\n")
    lines.append("| A | B | mean diff | Wilcoxon p | paired-t p | significant (0.05) |\n|---|---|---:|---:|---:|---|\n")
    for name_a, name_b in pairs:
        if len(results[name_a]) != len(results[name_b]) or len(results[name_a]) < 2:
            continue
        sig = compare_paired_results(results[name_a], results[name_b], metric_name="accuracy")
        lines.append(f"| {name_a} | {name_b} | {sig.mean_diff:+.3f} | {sig.wilcoxon_p} | {sig.paired_t_p} | {sig.significant_at_05} |\n")

    out_path = ROOT / "docs" / "_walk_forward_results.md"
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
