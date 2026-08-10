# Experiment Protocol

Every number in this document comes from an actual run of the scripts named in each
section (`scripts/run_walk_forward_experiment.py`,
`scripts/run_regime_calibration_xai_eval.py`, `scripts/run_baseline_comparison.py`). Where
a claim in the build brief has not been run, this document says "Not yet evaluated"
explicitly rather than omitting it.

## Method

- **Chronological, never shuffled, splits.** Every split in this project (single train/
  val/test and walk-forward folds) is verified via
  `emafrs.features.leakage_audit.check_chronological_split` — see
  `tests/unit/test_walk_forward.py` and `tests/integration/test_dataset_and_models.py`.
- **Walk-forward validation**: `emafrs.experiments.walk_forward.generate_walk_forward_folds`
  produces expanding-window folds (training data grows with each fold; test windows are
  non-overlapping and strictly increasing in time).
- **Regime model fit only on each fold's training window**, never on validation/test dates
  — see `dataset.py`'s `explicit_split` parameter and the leakage-boundary regression test.
- **Multiple seeds** for every neural (RAMA/static fusion) result: 3 seeds per fold/config.
- **Statistical significance**: paired Wilcoxon signed-rank test and paired t-test across
  folds (`emafrs.experiments.significance.compare_paired_results`), 95% confidence
  intervals via the t-distribution (appropriate for n=5 folds, not assumed-normal z-CIs).

## Scope limitation, disclosed up front

A genuine multi-fold walk-forward study needs every modality under comparison to be
available across ALL folds. In this reconstruction:
- **Financial** features: available for the full multi-year history — walk-forward-able.
- **Microstructure** (MMIE) proxies: yfinance only exposes 1-minute bars for the trailing
  7 days regardless of account tier (`docs/LIMITATIONS.md`) — genuinely cannot be walked
  forward across historical folds without fabricating "as of" availability for dates that
  never had real intraday data.
- **Text** (ASEM): not populated at all (no NewsAPI/YouTube key configured as of this
  experiment run).

Consequently: **walk-forward validation (below) uses financial-only features** across
Model A, B, and a single-modality neural baseline. **The multimodal RAMA-vs-static
comparison remains the single train/val/test split** from task #8/#9 (financial +
microstructure, ~5-7 real trading days of microstructure coverage). This is disclosed
here, not glossed over — a full multi-fold multimodal study is listed under "Not yet
evaluated" below, pending either a real intraday data source with deeper history or enough
accumulated live-pipeline history.

## Walk-forward results (real run, SPY, 2016-08-10 to 2026-08-10, 5 expanding-window folds)

Reproduce: `python scripts/run_walk_forward_experiment.py`

| Model | n_folds | Mean Accuracy | 95% CI | Per-fold accuracy |
|---|---:|---:|---|---|
| A: opaque XGBoost | 5 | 0.384 | [0.349, 0.419] | [0.363, 0.355, 0.382, 0.426, 0.393] |
| B: explainable logistic regression | 5 | 0.390 | [0.374, 0.406] | [0.371, 0.402, 0.386, 0.390, 0.401] |
| NN (single-modality fusion, 3 seeds/fold) | 5 | 0.402 | [0.379, 0.426] | [0.402, 0.400, 0.373, 0.424, 0.413] |

**Paired significance** (Wilcoxon signed-rank / paired t-test across the 5 folds):

| Comparison | Mean diff | Wilcoxon p | Paired-t p | Significant (α=0.05) |
|---|---:|---:|---:|---|
| A vs B | −0.006 | n/a (< 6 samples) | 0.656 | No |
| A vs NN | −0.019 | n/a (< 6 samples) | 0.164 | No |
| B vs NN | −0.012 | n/a (< 6 samples) | 0.256 | No |

**Finding, stated plainly**: none of the three models statistically significantly
outperforms the others at 1-day-ahead SPY direction prediction from financial/technical
features alone (all overlapping 95% CIs, all p > 0.16). Accuracy hovers around 38-40%
across a 3-class (bearish/neutral/bullish) problem where a class-frequency baseline would
land somewhere near 33-38% depending on the fold's label balance — i.e., this task is hard
and none of these models meaningfully solve it from financial features alone at a 1-day
horizon. This is consistent with market-efficiency expectations and is reported as a
genuine negative/null result, not adjusted or explained away. Wilcoxon could not be
computed (needs ≥6 paired samples; only 5 folds were run) — the paired t-test is the only
significance test available at this sample size, itself a limitation of only running 5 folds.

## Regime-wise performance, calibration, and XAI evaluation (real run, SPY, single split)

Reproduce: `python scripts/run_regime_calibration_xai_eval.py`. Model: RAMA
(financial+microstructure), single split, test accuracy=0.425, macro F1=0.259.

### Regime-wise performance

| Regime | N | Accuracy | Macro F1 | Mean fusion weight (financial) |
|---|---:|---:|---:|---:|
| 1 (Low) | 70 | 0.400 | 0.212 | 0.510 |
| 2 (Medium) | 53 | 0.434 | 0.258 | 0.500 |
| 3 (High) | 37 | 0.459 | 0.301 | 0.521 |

Regime 4 (Extreme) had fewer than 5 test-set observations and is excluded (too small a
sample for a meaningful per-regime metric, not silently zero-filled). Accuracy rises
mildly with regime severity but F1 stays low throughout, and — consistent with the
finding already logged in `docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md` row #19 — the
mean financial-modality fusion weight stays close to 0.50-0.52 across all three observed
regimes: RAMA's gate is not meaningfully regime-adaptive in this particular trained run,
even though the underlying mechanism is proven to be capable of it
(`tests/unit/test_fusion_models.py::test_rama_weights_differ_across_regimes`).

### Calibration

| | Brier score | ECE |
|---|---:|---:|
| Before regime temperature scaling | 0.660 | 0.019 |
| After regime temperature scaling | 0.673 | 0.049 |

Fitted temperatures by regime: `{1: 0.90, 2: 0.47, 3: 0.35, 4: 0.48}`.

**Finding, stated plainly**: regime-dependent temperature scaling made calibration
slightly WORSE on this test split, not better — the opposite of the intended effect. The
most likely cause is a small per-regime validation sample (~160 validation rows split
across 4 regimes, roughly 40 rows/regime) used to fit 4 independent temperatures via
LBFGS, which is prone to overfitting the validation split's own noise rather than learning
a temperature that generalizes to test. This is reported as a genuine limitation of this
single run, not corrected after the fact by re-tuning until it looked better — a larger
validation set or a regularized/shrinkage temperature fit (e.g. partial pooling toward a
single global temperature) is the natural next step, tracked under "Not yet evaluated."

### XAI evaluation (30-sample subset of the test set; full method in `docs/XAI_EVALUATION.md`)

| Metric | Result |
|---|---|
| Faithfulness (top-5 IG-attributed feature removal vs. random-5 removal) | −0.001 |
| Stability (attribution consistency under small input noise) | 0.946 |
| Sparsity (median # features for 80% attribution coverage, of 26 total) | 8.0 |
| Counterfactual validity rate | 0.80 (12/15) |

**Finding, stated plainly**: faithfulness is statistically indistinguishable from zero —
removing the top-5 Integrated-Gradients-attributed features degrades this model's
prediction about the same amount as removing 5 random features. This does not mean IG is
"broken" (its completeness axiom holds exactly, per `tests/unit/test_integrated_gradients.py`)
but it does mean, on this trained model, IG's top-ranked features are not more
functionally load-bearing than average — a real, disclosed weakness of this particular
model/attribution pairing, not evidence that faithfulness evaluation itself failed.
Stability and counterfactual validity are both reasonably strong.

## Not yet evaluated

- Multi-fold walk-forward for the multimodal RAMA-vs-static comparison (blocked on
  microstructure/text data depth — see Scope limitation above).
- Multiple tickers (only SPY has been run end-to-end through walk-forward and regime/
  calibration/XAI evaluation; single-split baseline comparison has also been run on SPY
  only via `scripts/run_baseline_comparison.py`).
- Full ablation grid A-K from the build brief (see `docs/ABLATION_STUDY.md` for exactly
  which rows are covered vs. not).
- Cross-dataset / cross-market external validation.
- Regularized/shrinkage regime-temperature fitting (flagged above as the natural fix for
  the calibration-worsening finding).
