# XAI Evaluation

Quantitative evaluation of the explainability layer (`src/emafrs/xai/`), per build brief
Section 21: XAI is evaluated as a measurable property of the trained model, not presented
only as visualization. Method implementations: `src/emafrs/xai/evaluation.py`. Real run:
`scripts/run_regime_calibration_xai_eval.py` (30-sample subset of the SPY single-split
test set, RAMA financial+microstructure model).

## Completeness (Integrated Gradients' own theoretical guarantee)

IG's completeness axiom states attributions must sum to `f(x) - f(baseline)`. This is
checked directly against real trained-model output, not merely assumed because Captum is
a trusted library (a wrapper bug could silently break it regardless of the underlying
library's correctness):

`tests/unit/test_integrated_gradients.py::test_ig_satisfies_completeness_axiom_*` — residual
`< 1e-3` for both StaticFusionModel and RAMAFusionModel. **Passing.**

## Faithfulness

**Method**: for each sampled test row, remove (zero) the top-5 IG-attributed features and
measure the drop in the model's predicted probability for its own predicted class, minus
the mean drop from removing 5 RANDOM features (same count). A faithful attribution should
degrade the prediction more than an equal-size random ablation — there is no principled
absolute threshold, so this is always reported relative to that random baseline
(`emafrs.xai.evaluation.faithfulness_score`).

**Result across three real runs, increasing sample size to test whether this is noise**:

| Run | n | Faithfulness |
|---|---:|---:|
| Original | 15 | −0.001 |
| Re-run, same n (different data window) | 15 | −0.004 |
| Re-run, 4× sample size | 60 | −0.0004 |

All three are statistically indistinguishable from zero, and **the largest sample (n=60)
is not farther from zero than the smaller ones — if anything, closer**. This directly
tests and rules out one of the three "plausible contributing factors" originally listed
below (small sample size / noise): a real effect that was actually nonzero-but-noisy at
n=15 would be expected to move away from zero, not toward it, as n grows. It did not.

**Honest interpretation**: on this trained RAMA model, IG's top-ranked features are NOT
more functionally load-bearing to the prediction than a random same-size feature subset —
now a reproducible finding across three independent measurements, not a single run's
statistical fluke. Remaining (still unconfirmed, but now better-supported by elimination)
contributing factors:
- The model's small size (hidden_dim=32, one hidden layer) and modest single-split
  training may produce a relatively flat/diffuse decision function where no small feature
  subset dominates. Consistent with the sparsity result below (median 8-9 of 26 features
  needed for 80% coverage — attribution mass is fairly spread out, not concentrated).
- Gradient-based attribution (IG) can diverge from removal-based (occlusion) importance
  when a model's response to a feature is highly nonlinear or saturates — a known,
  documented tension in the XAI literature, not unique to this implementation. This
  remains untested here (would require an occlusion-based attribution method to compare
  directly, not yet implemented — see "Not yet evaluated").
- ~~Small sample size~~ — tested directly above (n=15 → n=60) and ruled out as the
  explanation; retained here only to show the elimination, not as an open hypothesis.

This finding argues for caution in how the dashboard (once built) presents IG attributions
— as "model-attributed evidence" (the wording already enforced in `xai/nlg.py`), never as
"the reason," which this result underlines is not merely a legal hedge but an empirically
supported and now-reproduced caveat for this specific trained model.

## Stability

**Method**: for each sampled row, add small Gaussian noise (std=0.05 in the model's
z-scored feature space) 8 times and compute the Spearman rank correlation between the
original prediction-probability-vector response and each perturbed response
(`emafrs.xai.evaluation.stability_score`).

**Result**: 0.946 (n=15, original run); **0.970** (n=60, re-run) — consistently high
stability across both sample sizes.

**Interpretation**: the model's output — and by extension attributions computed from it —
does not flip wildly under small input perturbations. This is a genuinely positive
finding, reported alongside the faithfulness weakness above rather than cherry-picked
instead of it.

## Sparsity

**Method**: minimum number of (ranked by \|attribution\|) features whose cumulative
attribution magnitude reaches 80% of the total, per sampled row
(`emafrs.xai.evaluation.sparsity_at_coverage`).

**Result**: median 8 features (n=15, original run); median **9 features** (n=60, re-run),
of 26 total financial+microstructure features — moderately sparse and stable across
sample sizes; explanations are not spread perfectly evenly across all 26 features, but
also not dominated by 1-2 features alone.

## Counterfactual validity

**Method**: for each sampled row, `emafrs.xai.counterfactual.search_all_feature_counterfactuals`
grid-searches each financial feature (holding microstructure/regime fixed) for the minimal
perturbation (within ±6 standard deviations in z-scored space) that flips the model's
predicted class.

**Result (n=15)**: **80% validity rate** (12/15 samples had at least one feature whose
perturbation flipped the prediction within the search range); mean minimal \|delta\| among
flips ≈ **2.77** standard deviations.

**Interpretation**: for most sampled rows, a counterfactual explanation genuinely exists
and was found empirically (not merely asserted) — but the typical minimal change needed is
large (~2.8 std devs), meaning this model's decisions are not highly sensitive to small
realistic perturbations of any single feature. This is consistent with the faithfulness
finding above (no single feature dominates the decision).

## Attention/fusion-weight explanation (RAMA)

RAMA's per-row modality fusion weights are exposed directly as part of every forward pass
(`RAMAFusionModel.forward` returns `weights`) and are real softmax outputs, not a
post-hoc approximation — this is the one explanation channel in this system that is exact
by construction rather than estimated. Per `docs/EXPERIMENT_PROTOCOL.md`'s regime-wise
table, these weights stayed close to 0.50-0.52 (financial) across observed regimes in this
particular trained run — i.e., the attention explanation itself is honestly reporting that
this model did not learn strongly regime-differentiated fusion, which is a real (if
unflattering) property of the explanation being surfaced faithfully, not a flaw in the
explanation mechanism.

## Attention ≠ causality (explicit, per project-wide rule)

Per `docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md` and the project's anti-overclaiming rule:
none of the above — IG attribution, RAMA fusion weights, or counterfactual deltas — are
causal claims about the underlying market. They describe this specific trained model's
sensitivity to its own inputs. `xai/nlg.py`'s generated text enforces this wording
("model-attributed evidence," "not a causal claim") directly in the template, checked by
`tests/unit/test_nlg.py::test_generate_explanation_includes_all_provided_sections`.

## Not yet evaluated

- Occlusion-based (removal-based) attribution as a direct cross-check against IG's
  gradient-based attribution specifically for the faithfulness question — the
  larger-sample test above ruled out noise as the explanation for faithfulness≈0, leaving
  "gradient-vs-removal attribution divergence" as the strongest remaining untested
  hypothesis. Not yet implemented.
- The same battery on Model C (static fusion) for a head-to-head XAI-quality comparison
  against Model D (RAMA) — currently only RAMA has been evaluated.
- SHAP-based attribution as a cross-check against Integrated Gradients (both are installed
  and available; only IG has been run through the full evaluation battery so far).
- Human/stakeholder evaluation of explanation usefulness (build brief Section 3: "if a
  human/stakeholder evaluation can reasonably be conducted, design one") — no user study
  has been conducted; this remains a measurable-property evaluation only, not evidence of
  actual human trust or usefulness, and is described as such throughout this document.
  **RESULT NOT AVAILABLE** — no human subjects, IRB protocol, or user-study
  infrastructure exists in this project; fabricating trust/usefulness scores here would
  violate this project's no-fabrication rule. See the trustworthiness audit
  (`docs/TRUSTWORTHINESS_AUDIT.md`) for the full accounting of every capability this
  project does not have.
