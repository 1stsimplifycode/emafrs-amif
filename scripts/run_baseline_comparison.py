"""Trains and evaluates Models A-D on a single chronological split for one ticker.

This is a SINGLE split, not the full walk-forward/multi-seed rigor required for the final
research claims (that's docs/EXPERIMENT_PROTOCOL.md, task #10) -- this script exists to
prove the end-to-end pipeline (data -> features -> regime -> fusion -> evaluation) actually
works on real data and to get a first, honestly-labeled look at the numbers.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from emafrs.models.baselines import ModelAOpaqueXGBoost, ModelBExplainableLogReg, evaluate_predictions, encode_labels
from emafrs.models.dataset import build_daily_dataset
from emafrs.models.train import train_fusion_model


def main(ticker: str = "SPY"):
    print(f"Building dataset for {ticker}...")
    bundle = build_daily_dataset(ticker, period="5y", horizon=1, deadzone=0.0025)

    print(f"n_train={len(bundle.train_idx)} n_val={len(bundle.val_idx)} n_test={len(bundle.test_idx)}")
    print(f"financial features: {len(bundle.financial_columns)}, regime cols: {bundle.regime_columns}")
    print(f"microstructure cols: {len(bundle.microstructure_columns)}")
    print(f"train label distribution: {bundle.labels.loc[bundle.train_idx].value_counts().to_dict()}")
    print(f"test label distribution: {bundle.labels.loc[bundle.test_idx].value_counts().to_dict()}")
    print()

    X_train = bundle.features.loc[bundle.train_idx, bundle.financial_columns]
    X_test = bundle.features.loc[bundle.test_idx, bundle.financial_columns]
    y_train = bundle.labels.loc[bundle.train_idx]
    y_test = bundle.labels.loc[bundle.test_idx]

    print("Training Model A (opaque XGBoost, financial-only)...")
    model_a = ModelAOpaqueXGBoost().fit(X_train, y_train)
    proba_a = model_a.predict_proba(X_test)
    result_a = evaluate_predictions("A_opaque_xgboost", encode_labels(y_test), proba_a.argmax(axis=1), proba_a)

    print("Training Model B (explainable logistic regression, financial-only)...")
    model_b = ModelBExplainableLogReg().fit(X_train, y_train)
    proba_b = model_b.predict_proba(X_test)
    result_b = evaluate_predictions("B_explainable_logreg", encode_labels(y_test), proba_b.argmax(axis=1), proba_b)

    modality_cols = {"financial": bundle.financial_columns}
    if len(bundle.microstructure_columns) > 1:  # more than just the availability flag
        modality_cols["microstructure"] = bundle.microstructure_columns

    print("Training Model C (static multimodal fusion)...")
    _, result_c, weights_c = train_fusion_model(bundle, "static", modality_cols)

    print("Training Model D (RAMA -- regime-adaptive multimodal attention)...")
    _, result_d, weights_d = train_fusion_model(bundle, "rama", modality_cols)

    print()
    print(f"{'Model':<28s} {'Accuracy':>10s} {'MacroF1':>10s} {'AUC(ovr)':>10s} {'N':>6s}")
    for r in [result_a, result_b, result_c, result_d]:
        auc_str = f"{r.auc_ovr:.3f}" if r.auc_ovr is not None else "n/a"
        print(f"{r.model_name:<28s} {r.accuracy:>10.3f} {r.macro_f1:>10.3f} {auc_str:>10s} {r.n_test:>6d}")

    print()
    print("Model C (static) mean fusion weights on test set:", weights_c.mean(axis=0).round(3), list(modality_cols))
    print("Model D (RAMA) mean fusion weights on test set:  ", weights_d.mean(axis=0).round(3), list(modality_cols))
    print("Model D (RAMA) fusion weight STD across test rows:", weights_d.std(axis=0).round(3),
          "(nonzero std -> weights actually vary by regime, unlike Model C's fixed weights)")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    main(ticker)
