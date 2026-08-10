"""Regime-wise performance, calibration (regime-dependent temperature scaling), and
quantitative XAI evaluation (faithfulness, stability, sparsity, counterfactual validity)
on the RAMA financial+microstructure model from the single-split run (task #8/#9).

Real numbers only -- written to docs/_regime_calibration_xai_results.md.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import torch

from emafrs.models.calibration import (
    apply_regime_temperature,
    brier_score,
    evaluate_regime_calibration,
    expected_calibration_error,
    fit_global_temperature,
    fit_regime_temperature,
    fit_regime_temperature_shrinkage,
)
from emafrs.models.dataset import build_daily_dataset
from emafrs.models.train import train_fusion_model
from emafrs.regime.hmm_regime import REGIME_NAMES
from emafrs.xai.evaluation import counterfactual_validity_rate, faithfulness_score, sparsity_at_coverage, stability_score
from emafrs.xai.integrated_gradients import compute_integrated_gradients

ROOT = Path(__file__).resolve().parents[1]
TICKER = "SPY"


def main():
    print(f"Building single-split dataset for {TICKER} (financial + microstructure)...")
    bundle = build_daily_dataset(TICKER, period="5y", horizon=1, deadzone=0.0025)
    modality_cols = {"financial": bundle.financial_columns}
    if len(bundle.microstructure_columns) > 1:
        modality_cols["microstructure"] = bundle.microstructure_columns

    print("Training Model D (RAMA)...")
    model, result, _ = train_fusion_model(bundle, "rama", modality_cols, epochs=150, patience=15, seed=42)
    print(f"Test accuracy={result.accuracy:.3f} macro_f1={result.macro_f1:.3f}")

    # ---- Regime-wise performance ----
    def _tensors(idx):
        return {name: torch.tensor(bundle.features.loc[idx, cols].to_numpy(), dtype=torch.float32)
                for name, cols in modality_cols.items()}

    from emafrs.models.baselines import encode_labels

    X_test = _tensors(bundle.test_idx)
    regime_test = torch.tensor(bundle.features.loc[bundle.test_idx, bundle.regime_columns].to_numpy(), dtype=torch.float32)
    y_test_idx = encode_labels(bundle.labels.loc[bundle.test_idx])
    regime_argmax = regime_test.argmax(dim=1).numpy() + 1  # 1-indexed regime label

    model.eval()
    with torch.no_grad():
        logits, weights = model(X_test, regime_test)
        proba = torch.softmax(logits, dim=1).numpy()
    pred_idx = proba.argmax(axis=1)

    print("\n=== Regime-wise performance (test set) ===")
    regime_lines = ["| Regime | N | Accuracy | Macro F1 | Mean fusion weight (financial) |\n|---|---:|---:|---:|---:|\n"]
    from sklearn.metrics import accuracy_score, f1_score
    for r in sorted(set(regime_argmax.tolist())):
        mask = regime_argmax == r
        if mask.sum() < 5:
            continue
        acc = accuracy_score(y_test_idx[mask], pred_idx[mask])
        f1 = f1_score(y_test_idx[mask], pred_idx[mask], average="macro", zero_division=0)
        mean_fin_weight = weights[mask, 0].mean().item()
        name = REGIME_NAMES.get(int(r), str(r))
        print(f"  Regime {r} ({name}): n={mask.sum()} acc={acc:.3f} f1={f1:.3f} mean_financial_weight={mean_fin_weight:.3f}")
        regime_lines.append(f"| {r} ({name}) | {mask.sum()} | {acc:.3f} | {f1:.3f} | {mean_fin_weight:.3f} |\n")

    # ---- Calibration ----
    print("\n=== Calibration (regime-dependent temperature scaling) ===")
    X_val = _tensors(bundle.val_idx)
    regime_val = torch.tensor(bundle.features.loc[bundle.val_idx, bundle.regime_columns].to_numpy(), dtype=torch.float32)
    y_val_idx = torch.tensor(encode_labels(bundle.labels.loc[bundle.val_idx]), dtype=torch.long)
    regime_val_argmax = regime_val.argmax(dim=1)

    with torch.no_grad():
        val_logits, _ = model(X_val, regime_val)
    n_regimes = len(bundle.regime_columns)
    regime_test_argmax_t = regime_test.argmax(dim=1)
    brier_before = brier_score(proba, y_test_idx)
    ece_before = expected_calibration_error(proba, y_test_idx)

    # Naive per-regime temperature (the ORIGINAL implementation) -- documented in
    # docs/EXPERIMENT_PROTOCOL.md as making test-set calibration WORSE, hypothesized
    # cause: 4 fully independent temperatures fit on ~40 val rows/regime overfit
    # validation-split noise. Two corrections are compared against it below on the SAME
    # real test set: a single global temperature (no per-regime freedom at all), and a
    # shrinkage/partial-pooling fit (per-regime temperatures regularized toward a jointly
    # fit global temperature).
    temp_naive = fit_regime_temperature(val_logits, y_val_idx, regime_val_argmax, n_regimes=n_regimes)
    temp_global_scalar = fit_global_temperature(val_logits, y_val_idx)
    temp_global = temp_global_scalar.expand(n_regimes)
    temp_shrink = fit_regime_temperature_shrinkage(val_logits, y_val_idx, regime_val_argmax, n_regimes=n_regimes, shrinkage_weight=2.0)

    calibration_variants = {
        "naive_per_regime": temp_naive,
        "global_only": temp_global,
        "shrinkage_per_regime (weight=2.0)": temp_shrink,
    }

    print(f"  Before calibration: Brier={brier_before:.4f} ECE={ece_before:.4f}")
    calib_lines = ["\n## Calibration\n\n", f"Before calibration: Brier={brier_before:.4f}, ECE={ece_before:.4f}.\n\n",
                   "| Variant | Brier | ECE | Fitted temperature(s) |\n|---|---:|---:|---|\n"]
    best_variant, best_ece = None, ece_before
    for name, temp in calibration_variants.items():
        proba_calibrated = apply_regime_temperature(logits, regime_test_argmax_t, temp).detach().numpy()
        brier_after = brier_score(proba_calibrated, y_test_idx)
        ece_after = expected_calibration_error(proba_calibrated, y_test_idx)
        temp_str = ", ".join(f"{t:.3f}" for t in temp.tolist())
        print(f"  [{name}] Brier={brier_after:.4f} ECE={ece_after:.4f} temp(s)=[{temp_str}]")
        calib_lines.append(f"| {name} | {brier_after:.4f} | {ece_after:.4f} | [{temp_str}] |\n")
        if ece_after < best_ece:
            best_variant, best_ece = name, ece_after

    print(f"  Best-on-test variant by ECE: {best_variant or 'NONE -- no calibration variant beat uncalibrated'} "
          f"(uncalibrated ECE={ece_before:.4f})")
    calib_lines.append(f"\nBest-on-test variant by ECE: **{best_variant or 'NONE -- no calibration variant improved on uncalibrated'}** "
                        f"(uncalibrated ECE={ece_before:.4f}).\n")

    # Use the naive per-regime temperature downstream (regime-wise report, doc numbers)
    # for continuity with the previously-published per-regime breakdown; the honest
    # cross-variant comparison above is the actual correction-and-re-evaluation result.
    temperature = temp_naive
    proba_calibrated = apply_regime_temperature(logits, regime_test_argmax_t, temperature).detach().numpy()
    brier_after = brier_score(proba_calibrated, y_test_idx)
    ece_after = expected_calibration_error(proba_calibrated, y_test_idx)

    report = evaluate_regime_calibration(
        proba_calibrated, y_test_idx, (regime_test_argmax_t + 1).numpy(), temperature.numpy()
    )

    # ---- XAI evaluation ----
    print("\n=== XAI evaluation (sampled test rows) ===")
    n_sample = min(60, len(bundle.test_idx))
    sample_idx = np.random.default_rng(0).choice(len(bundle.test_idx), size=n_sample, replace=False)

    def predict_proba_flat(x_flat: np.ndarray) -> np.ndarray:
        n_fin = len(bundle.financial_columns)
        inp = {"financial": torch.tensor(x_flat[:n_fin], dtype=torch.float32).unsqueeze(0)}
        if "microstructure" in modality_cols:
            inp["microstructure"] = torch.tensor(x_flat[n_fin:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logit, _ = model(inp, regime_test[0:1])
            return torch.softmax(logit, dim=1)[0].numpy()

    def predict_class_flat(x_flat: np.ndarray) -> int:
        return int(predict_proba_flat(x_flat).argmax())

    all_feature_names = list(bundle.financial_columns) + (list(bundle.microstructure_columns) if "microstructure" in modality_cols else [])

    faithfulness_scores, stability_scores, sparsities = [], [], []
    for i in sample_idx[:60]:  # bumped from 15 -> 60 to test whether the faithfulness ~0
        # finding was a real effect or noise from a small evaluation sample (a plausible
        # contributing factor flagged, but not tested, in docs/XAI_EVALUATION.md)
        date = bundle.test_idx[i]
        row = bundle.features.loc[date, all_feature_names].to_numpy(dtype=np.float32)
        target_class = int(pred_idx[i])

        attrs = compute_integrated_gradients(
            model, {k: v[i:i+1] for k, v in X_test.items()}, list(modality_cols),
            target_class=target_class, regime_probs=regime_test[i:i+1], n_steps=30,
        )
        flat_attr = np.concatenate([attrs[m][0] for m in modality_cols])

        faithfulness_scores.append(faithfulness_score(predict_proba_flat, row, flat_attr, target_class, top_k=5))
        stability_scores.append(stability_score(lambda x: predict_proba_flat(x), row, n_perturbations=8, noise_std=0.05))
        sparsities.append(sparsity_at_coverage(flat_attr, coverage=0.8))

    print(f"  Faithfulness (top-5 vs random-5 removal, mean over {len(faithfulness_scores)} samples): {np.mean(faithfulness_scores):.4f}")
    print(f"  Stability (mean probability-vector cosine-like score over perturbations): {np.mean(stability_scores):.4f}")
    print(f"  Sparsity (median #features for 80% attribution coverage, out of {len(all_feature_names)}): {np.median(sparsities):.1f}")

    print("  Running counterfactual validity search over sampled rows (financial modality only)...")
    X_cf = np.stack([bundle.features.loc[bundle.test_idx[i], bundle.financial_columns].to_numpy(dtype=np.float32) for i in sample_idx[:15]])

    def predict_fn_financial_only(x_fin: np.ndarray) -> int:
        with torch.no_grad():
            inp = {"financial": torch.tensor(x_fin, dtype=torch.float32).unsqueeze(0)}
            if "microstructure" in modality_cols:
                inp["microstructure"] = X_test["microstructure"][0:1]
            logit, _ = model(inp, regime_test[0:1])
            return int(torch.softmax(logit, dim=1).argmax(dim=1).item())

    cf_report = counterfactual_validity_rate(
        predict_fn_financial_only, X_cf, bundle.financial_columns, search_range=(-6.0, 6.0), n_steps=31
    )
    print(f"  Counterfactual validity rate: {cf_report.validity_rate:.2f} ({cf_report.n_flipped}/{cf_report.n_samples}), "
          f"mean min |delta|={cf_report.mean_min_abs_delta}")

    # ---- Write results doc ----
    lines = ["# Regime-wise performance, calibration, and XAI evaluation (real run)\n\n"]
    lines.append(f"Ticker: {TICKER}, single split, RAMA model (financial+microstructure), test accuracy={result.accuracy:.3f}, macro F1={result.macro_f1:.3f}.\n\n")
    lines.append("## Regime-wise performance\n\n")
    lines += regime_lines
    lines += calib_lines
    lines.append(f"\nNaive-per-regime detail -- Brier after={brier_after:.4f}, ECE after={ece_after:.4f}, "
                  f"fitted temperature by regime: {report.temperature_by_regime}\n")
    lines.append(f"Per-regime Brier: {report.per_regime_brier}\nPer-regime ECE: {report.per_regime_ece}\n")
    lines.append("\n## XAI evaluation\n\n")
    lines.append(f"- Faithfulness (top-5 vs random-5 removal degradation, n={len(faithfulness_scores)}): {np.mean(faithfulness_scores):.4f}\n")
    lines.append(f"- Stability (mean attribution similarity under input noise, n={len(stability_scores)}): {np.mean(stability_scores):.4f}\n")
    lines.append(f"- Sparsity (median features for 80% coverage, out of {len(all_feature_names)}): {np.median(sparsities):.1f}\n")
    lines.append(f"- Counterfactual validity rate: {cf_report.validity_rate:.2f} ({cf_report.n_flipped}/{cf_report.n_samples})\n")

    out_path = ROOT / "docs" / "_regime_calibration_xai_results.md"
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
