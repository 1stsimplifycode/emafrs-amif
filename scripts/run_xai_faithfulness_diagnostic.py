"""Deep diagnostic of the IG faithfulness ~0 finding (docs/XAI_EVALUATION.md), run during
the trustworthiness-audit remediation pass. Investigates, on the SAME real trained RAMA
model and SAME real test set used throughout this project:

  (A) baseline/reference choice -- current zero-in-zscored-space baseline vs a
      random-real-training-sample baseline
  (B) gradient saturation -- gradient norm along the IG integration path
  (C) alternative attribution methods -- Gradient x Input (still gradient-based, no path
      integration) and Occlusion (no gradient dependency at all)
  (D) full deletion/insertion/comprehensiveness/sufficiency curves, not just the single
      top-5-vs-random-5 point already reported
  (E) modality dominance -- does one modality capture most attribution mass?

Real numbers only -- written to docs/_xai_diagnostic_results.md.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import torch

from emafrs.models.dataset import build_daily_dataset
from emafrs.models.train import train_fusion_model
from emafrs.xai.alternative_attributions import compute_gradient_x_input, compute_occlusion
from emafrs.xai.evaluation import (
    comprehensiveness,
    deletion_auc,
    faithfulness_score,
    insertion_auc,
    sufficiency,
)
from emafrs.xai.integrated_gradients import compute_integrated_gradients

ROOT = Path(__file__).resolve().parents[1]
TICKER = "SPY"
N_SAMPLES = 60


def main():
    print(f"Building single-split dataset for {TICKER} (financial + microstructure)...")
    bundle = build_daily_dataset(TICKER, period="5y", horizon=1, deadzone=0.0025)
    modality_cols = {"financial": bundle.financial_columns}
    if len(bundle.microstructure_columns) > 1:
        modality_cols["microstructure"] = bundle.microstructure_columns
    all_feature_names = list(bundle.financial_columns) + list(bundle.microstructure_columns)

    print("Training Model D (RAMA)...")
    model, result, _ = train_fusion_model(bundle, "rama", modality_cols, epochs=150, patience=15, seed=42)
    print(f"Test accuracy={result.accuracy:.3f} macro_f1={result.macro_f1:.3f}")

    def _tensors(idx):
        return {name: torch.tensor(bundle.features.loc[idx, cols].to_numpy(), dtype=torch.float32)
                for name, cols in modality_cols.items()}

    from emafrs.models.baselines import encode_labels

    X_test = _tensors(bundle.test_idx)
    X_train = _tensors(bundle.train_idx)
    regime_test = torch.tensor(bundle.features.loc[bundle.test_idx, bundle.regime_columns].to_numpy(), dtype=torch.float32)
    y_test_idx = encode_labels(bundle.labels.loc[bundle.test_idx])

    model.eval()
    with torch.no_grad():
        logits, _ = model(X_test, regime_test)
        proba = torch.softmax(logits, dim=1).numpy()
    pred_idx = proba.argmax(axis=1)

    def predict_proba_flat(x_flat: np.ndarray) -> np.ndarray:
        n_fin = len(bundle.financial_columns)
        inp = {"financial": torch.tensor(x_flat[:n_fin], dtype=torch.float32).unsqueeze(0)}
        if "microstructure" in modality_cols:
            inp["microstructure"] = torch.tensor(x_flat[n_fin:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logit, _ = model(inp, regime_test[0:1])
            return torch.softmax(logit, dim=1)[0].numpy()

    rng = np.random.default_rng(0)
    n_sample = min(N_SAMPLES, len(bundle.test_idx))
    sample_idx = rng.choice(len(bundle.test_idx), size=n_sample, replace=False)

    # ---- (A) + (C) + (D): compute all attribution methods + all faithfulness metrics ----
    print(f"\n=== Computing attributions (IG, Grad*Input, Occlusion) for {n_sample} samples ===")
    methods_results = {m: {"faithfulness": [], "deletion_auc": [], "insertion_auc": [],
                            "comprehensiveness": [], "sufficiency": []}
                        for m in ["ig_zero_baseline", "ig_random_sample_baseline", "grad_x_input", "occlusion"]}
    modality_mass = {"financial": [], "microstructure": []}

    n_fin = len(bundle.financial_columns)
    train_sample_rows = X_train["financial"][rng.choice(len(bundle.train_idx), size=n_sample, replace=True)]
    train_sample_micro = X_train.get("microstructure")
    if train_sample_micro is not None:
        train_sample_micro_rows = train_sample_micro[rng.choice(len(bundle.train_idx), size=n_sample, replace=True)]

    for j, i in enumerate(sample_idx):
        date = bundle.test_idx[i]
        row = bundle.features.loc[date, all_feature_names].to_numpy(dtype=np.float32)
        target_class = int(pred_idx[i])
        row_inputs = {k: v[i:i+1] for k, v in X_test.items()}

        # IG, zero baseline (the CURRENT default -- zero in z-scored space == the
        # rolling-window mean of the raw feature, not an arbitrary raw-scale zero)
        attrs_ig_zero = compute_integrated_gradients(
            model, row_inputs, list(modality_cols), target_class=target_class,
            regime_probs=regime_test[i:i+1], n_steps=30,
        )
        flat_ig_zero = np.concatenate([attrs_ig_zero[m][0] for m in modality_cols])

        # IG, random-real-training-sample baseline (Sundararajan et al. discuss
        # baseline choice as a real degree of freedom; a real training row is a
        # legitimate "typical" reference point, distinct from the all-zero/mean point)
        alt_baseline_tensors = {"financial": train_sample_rows[j:j+1]}
        if train_sample_micro is not None:
            alt_baseline_tensors["microstructure"] = train_sample_micro_rows[j:j+1]
        from captum.attr import IntegratedGradients
        from emafrs.xai.integrated_gradients import _PositionalWrapper
        is_rama = True
        wrapper = _PositionalWrapper(model, list(modality_cols), target_class, is_rama)
        wrapper.eval()
        ig = IntegratedGradients(wrapper)
        tensors = tuple(row_inputs[name].clone().detach().requires_grad_(True) for name in modality_cols)
        alt_baselines = tuple(alt_baseline_tensors[name].clone().detach() for name in modality_cols)
        tensors_full = tensors + (regime_test[i:i+1].clone().detach(),)
        alt_baselines_full = alt_baselines + (torch.full_like(regime_test[i:i+1], 1.0 / regime_test.shape[1]),)
        attrs_raw = ig.attribute(tensors_full, baselines=alt_baselines_full, n_steps=30)
        attrs_ig_alt = {name: attrs_raw[k].detach().numpy() for k, name in enumerate(modality_cols)}
        flat_ig_alt = np.concatenate([attrs_ig_alt[m][0] for m in modality_cols])

        # Gradient x Input
        attrs_gxi = compute_gradient_x_input(
            model, row_inputs, list(modality_cols), target_class=target_class, regime_probs=regime_test[i:i+1],
        )
        flat_gxi = np.concatenate([attrs_gxi[m][0] for m in modality_cols])

        # Occlusion
        attrs_occ = compute_occlusion(
            model, row_inputs, list(modality_cols), target_class=target_class, regime_probs=regime_test[i:i+1],
        )
        flat_occ = np.concatenate([attrs_occ[m][0] for m in modality_cols])

        for name, flat_attr in [
            ("ig_zero_baseline", flat_ig_zero), ("ig_random_sample_baseline", flat_ig_alt),
            ("grad_x_input", flat_gxi), ("occlusion", flat_occ),
        ]:
            methods_results[name]["faithfulness"].append(
                faithfulness_score(predict_proba_flat, row, flat_attr, target_class, top_k=5))
            methods_results[name]["deletion_auc"].append(
                deletion_auc(predict_proba_flat, row, flat_attr, target_class))
            methods_results[name]["insertion_auc"].append(
                insertion_auc(predict_proba_flat, row, flat_attr, target_class))
            methods_results[name]["comprehensiveness"].append(
                comprehensiveness(predict_proba_flat, row, flat_attr, target_class, top_k=5))
            methods_results[name]["sufficiency"].append(
                sufficiency(predict_proba_flat, row, flat_attr, target_class, top_k=5))

        # modality dominance (using the primary IG-zero-baseline attribution)
        fin_mass = np.abs(flat_ig_zero[:n_fin]).sum()
        micro_mass = np.abs(flat_ig_zero[n_fin:]).sum()
        total_mass = fin_mass + micro_mass
        if total_mass > 0:
            modality_mass["financial"].append(fin_mass / total_mass)
            modality_mass["microstructure"].append(micro_mass / total_mass)

        if (j + 1) % 15 == 0:
            print(f"  ...{j + 1}/{n_sample} samples done")

    # ---- (B) saturation check: gradient norm along the IG path for a few samples ----
    print("\n=== Saturation check: gradient norm along the IG integration path ===")
    from emafrs.xai.integrated_gradients import _PositionalWrapper as PW
    saturation_lines = []
    for i in sample_idx[:5]:
        date = bundle.test_idx[i]
        target_class = int(pred_idx[i])
        row_inputs = {k: v[i:i+1] for k, v in X_test.items()}
        wrapper = PW(model, list(modality_cols), int(pred_idx[i]), True)
        wrapper.eval()
        baseline_fin = torch.zeros_like(row_inputs["financial"])
        baseline_micro = torch.zeros_like(row_inputs.get("microstructure", row_inputs["financial"])) if "microstructure" in row_inputs else None
        grad_norms = []
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            fin_interp = (baseline_fin + alpha * (row_inputs["financial"] - baseline_fin)).clone().requires_grad_(True)
            call_tensors = [fin_interp]
            if baseline_micro is not None:
                micro_interp = (baseline_micro + alpha * (row_inputs["microstructure"] - baseline_micro)).clone().requires_grad_(True)
                call_tensors.append(micro_interp)
            else:
                micro_interp = None
            out = wrapper(*call_tensors, regime_test[i:i+1])
            out.backward()
            total_norm = fin_interp.grad.norm().item()
            if micro_interp is not None:
                total_norm += micro_interp.grad.norm().item()
            grad_norms.append(total_norm)
        saturation_lines.append(f"  sample {date.date()}: grad norms at alpha=[0,.25,.5,.75,1] = {[round(g, 4) for g in grad_norms]}")
        print(saturation_lines[-1])

    # ---- Summarize ----
    print("\n=== Summary: faithfulness metrics by attribution method ===")
    summary_lines = ["# XAI faithfulness diagnostic (real run)\n\n",
                      f"Model: RAMA (financial+microstructure), SPY, test accuracy={result.accuracy:.3f}, n_samples={n_sample}.\n\n",
                      "## Faithfulness metrics by attribution method\n\n",
                      "| Method | Faithfulness (top5 vs rand5) | Deletion AUC (lower=better) | Insertion AUC (higher=better) | Comprehensiveness | Sufficiency (lower=better) |\n",
                      "|---|---:|---:|---:|---:|---:|\n"]
    for method, metrics in methods_results.items():
        f_mean = np.mean(metrics["faithfulness"])
        d_mean = np.mean(metrics["deletion_auc"])
        i_mean = np.mean(metrics["insertion_auc"])
        c_mean = np.mean(metrics["comprehensiveness"])
        s_mean = np.mean(metrics["sufficiency"])
        print(f"  {method:28s} faithfulness={f_mean:+.4f}  deletion_auc={d_mean:.4f}  insertion_auc={i_mean:.4f}  "
              f"comprehensiveness={c_mean:+.4f}  sufficiency={s_mean:+.4f}")
        summary_lines.append(f"| {method} | {f_mean:+.4f} | {d_mean:.4f} | {i_mean:.4f} | {c_mean:+.4f} | {s_mean:+.4f} |\n")

    print("\n=== Modality dominance (IG zero-baseline attribution mass share) ===")
    fin_share = np.mean(modality_mass["financial"])
    micro_share = np.mean(modality_mass["microstructure"])
    print(f"  financial mean mass share: {fin_share:.3f}")
    print(f"  microstructure mean mass share: {micro_share:.3f}")
    summary_lines.append(f"\n## Modality dominance (IG zero-baseline attribution mass share, n={n_sample})\n\n")
    summary_lines.append(f"financial: {fin_share:.3f}, microstructure: {micro_share:.3f}\n")

    summary_lines.append("\n## Saturation check (gradient norm along the IG path, alpha=0..1, 5 samples)\n\n")
    summary_lines.append("\n".join(saturation_lines) + "\n")

    out_path = ROOT / "docs" / "_xai_diagnostic_results.md"
    out_path.write_text("".join(summary_lines), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
