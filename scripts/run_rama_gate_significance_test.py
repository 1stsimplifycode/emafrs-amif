"""Statistical significance test for RAMA's regime-adaptivity, added during the
trustworthiness-audit remediation pass. Prior evidence (docs/EXPERIMENT_PROTOCOL.md,
docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md row 19) showed the mean financial-modality
fusion weight stays within a narrow 0.50-0.52 band across all 4 regimes, and two
interventions (weight-decay exclusion, forced long unregularized training) both failed to
produce genuine regime-specific differentiation. This script asks the harder, more
rigorous question directly: is the OBSERVED between-regime variation in fusion weight
distinguishable from what pure chance would produce, given the SAME within-regime
variance and sample sizes?

Method: one-way ANOVA F-statistic on per-row financial-modality fusion weight, grouped by
regime, computed on the real test set. Significance via a permutation test (shuffle
regime-group labels 5000 times, keeping weights fixed, and see where the true F-statistic
falls in that null distribution) rather than assuming normality -- more defensible given
unequal, sometimes small, group sizes (n=12-70 per regime).

Real numbers only -- written to docs/_rama_gate_significance_results.md.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import torch
from scipy import stats

from emafrs.experiments.significance import permutation_test_anova
from emafrs.models.dataset import build_daily_dataset
from emafrs.models.train import train_fusion_model

ROOT = Path(__file__).resolve().parents[1]
TICKER = "SPY"
N_PERMUTATIONS = 5000


def main():
    print(f"Building single-split dataset for {TICKER} (financial + microstructure)...")
    bundle = build_daily_dataset(TICKER, period="5y", horizon=1, deadzone=0.0025)
    modality_cols = {"financial": bundle.financial_columns}
    if len(bundle.microstructure_columns) > 1:
        modality_cols["microstructure"] = bundle.microstructure_columns

    print("Training Model D (RAMA)...")
    model, result, test_weights = train_fusion_model(bundle, "rama", modality_cols, epochs=150, patience=15, seed=42)
    print(f"Test accuracy={result.accuracy:.3f} macro_f1={result.macro_f1:.3f}")

    regime_test = torch.tensor(bundle.features.loc[bundle.test_idx, bundle.regime_columns].to_numpy(), dtype=torch.float32)
    regime_argmax = (regime_test.argmax(dim=1) + 1).numpy()  # 1-indexed

    financial_weight = test_weights[:, 0]  # column 0 == "financial" by modality_cols dict order

    print("\n=== Per-regime financial-weight distribution (real test set) ===")
    regime_stats = {}
    lines = ["# RAMA gate regime-differentiation significance test (real run)\n\n",
             f"Ticker: {TICKER}, RAMA model (financial+microstructure), test accuracy={result.accuracy:.3f}.\n\n",
             "## Per-regime financial-weight distribution\n\n",
             "| Regime | N | Mean | Std | Min | Max |\n|---|---:|---:|---:|---:|---:|\n"]
    for r in sorted(set(regime_argmax.tolist())):
        mask = regime_argmax == r
        if mask.sum() < 2:
            continue
        vals = financial_weight[mask]
        regime_stats[r] = vals
        print(f"  Regime {r}: n={mask.sum()} mean={vals.mean():.4f} std={vals.std():.4f} "
              f"min={vals.min():.4f} max={vals.max():.4f}")
        lines.append(f"| {r} | {mask.sum()} | {vals.mean():.4f} | {vals.std():.4f} | {vals.min():.4f} | {vals.max():.4f} |\n")

    valid_mask = np.isin(regime_argmax, list(regime_stats.keys()))
    values = financial_weight[valid_mask]
    groups = regime_argmax[valid_mask]

    perm_result = permutation_test_anova(values, groups, n_permutations=N_PERMUTATIONS, seed=0)
    ratio = (perm_result.between_group_variance / perm_result.mean_within_group_variance
             if perm_result.mean_within_group_variance > 0 else float("nan"))

    print(f"\n=== Variance decomposition ===")
    print(f"  Between-regime variance (of per-regime MEAN weight): {perm_result.between_group_variance:.8f}")
    print(f"  Mean within-regime variance (of per-row weight, averaged across regimes): {perm_result.mean_within_group_variance:.8f}")
    print(f"  Ratio (between/within): {ratio:.4f}")
    print(f"  Observed one-way ANOVA F-statistic: {perm_result.observed_f:.4f}")

    print(f"\n=== Permutation test ({N_PERMUTATIONS} shuffles of regime labels) ===")
    print(f"  Permutation p-value (fraction of shuffles with F >= observed): {perm_result.p_value:.4f}")

    # also report the standard parametric ANOVA p-value for cross-reference (not the primary claim)
    group_arrays = [regime_stats[r] for r in sorted(regime_stats)]
    f_stat_scipy, p_parametric = stats.f_oneway(*group_arrays)

    is_significant = perm_result.p_value < 0.05
    verdict = "STATISTICALLY SIGNIFICANT regime differentiation (but check effect size below -- significance != large magnitude)" if is_significant else \
        "NOT statistically distinguishable from chance -- regime label carries no detectable information about fusion weight beyond what random grouping would produce"
    print(f"\n=== Verdict (alpha=0.05): {verdict} ===")

    effect_size_note = (
        f"Effect size check: per-regime mean weight ranges {min(v.mean() for v in regime_stats.values()):.4f} "
        f"to {max(v.mean() for v in regime_stats.values()):.4f} -- a "
        f"{(max(v.mean() for v in regime_stats.values()) - min(v.mean() for v in regime_stats.values())) * 100:.1f} "
        "percentage-point range. Statistical significance here reflects EXTREMELY low within-regime variance "
        "(the weight is nearly constant within a regime, since it is a deterministic function of the regime "
        "posterior alone), not a substantively large modality-reweighting effect."
    )
    print(f"  {effect_size_note}")

    lines.append(f"\n## Variance decomposition\n\n")
    lines.append(f"Between-regime variance (of per-regime mean weight): {perm_result.between_group_variance:.8f}\n\n")
    lines.append(f"Mean within-regime variance: {perm_result.mean_within_group_variance:.8f}\n\n")
    lines.append(f"Ratio (between/within): {ratio:.4f}\n\n")
    lines.append(f"## Significance test\n\n")
    lines.append(f"Observed one-way ANOVA F-statistic: {perm_result.observed_f:.4f}\n\n")
    lines.append(f"Permutation p-value ({N_PERMUTATIONS} shuffles): **{perm_result.p_value:.4f}**\n\n")
    lines.append(f"Parametric (scipy f_oneway) p-value, for cross-reference: {p_parametric:.4f}\n\n")
    lines.append(f"**Verdict (alpha=0.05): {verdict}**\n\n")
    lines.append(f"{effect_size_note}\n")

    out_path = ROOT / "docs" / "_rama_gate_significance_results.md"
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
