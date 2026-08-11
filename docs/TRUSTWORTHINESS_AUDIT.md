# Trustworthiness Audit: EMAFRS / AMIF

**Document structure note**: Sections A-L below are the original diagnostic audit.
Section M ("Remediation Pass") is a second pass that took every finding in A-L and asked,
for each one: is this fixable, and if so, was it actually fixed and re-verified? Section M
does not delete or silently rewrite A-L — where remediation changed a conclusion, A-L was
updated in place with an explicit note and a pointer to Section M's fuller account, so the
original finding, the correction attempt, and the final state are all still visible. Read
Section M first for the current, most-accurate picture; read A-L for how each finding was
originally reached.

A diagnostic, evidence-only audit of this project's real trustworthiness properties —
predictive performance, calibration, explainability/faithfulness, affective computing,
cross-modal consistency, missing-modality robustness, uncertainty/OOD, and human trust —
scoped to EMAFRS/AMIF exactly as built (per the reconstruction of `PROJECT DOCUMENT.pdf`),
not to a hypothetical system with capabilities this project does not have.

**Ground rule, applied throughout**: every claim below is either (a) backed by a real run
cited by file/script/line, (b) marked **RESULT NOT AVAILABLE** with the specific missing
capability named, or (c) marked as a root-caused, corrected, and re-evaluated finding with
the before/after numbers shown. Nothing is estimated, interpolated, or presented as
plausible-sounding filler. Where a correction was attempted and did NOT work, that is
reported as the finding — this document does not manufacture a positive conclusion.

---

## A. Executive summary

*(Original. See §M for what changed after remediation.)*

This system's predictive core (financial-only direction prediction) performs at a level
statistically indistinguishable from a class-frequency baseline (~38-42% accuracy on a
3-class problem), consistent with market-efficiency expectations and reported as a
genuine null result, not adjusted. Two mechanisms named in the source patent as central —
regime-adaptive fusion (RAMA) and Integrated-Gradients explainability — are **proven to
be mechanically correct** (unit-tested, axiom-verified) but **do not demonstrate their
intended value on this real trained model**: RAMA's gate stays near-uniform across
regimes (root-caused below, not a bug — **UPDATED in §M: this is now known to be
statistically significant but small in magnitude, not "no effect"**), and IG's
attributions are no more load-bearing than random feature removal (reproduced across
three independent runs at increasing sample size, ruling out noise as the explanation —
**UPDATED in §M: now reproduced across FOUR independent attribution methods**). Regime-dependent calibration was found
to make things worse, then — during this audit — a real, previously-undiagnosed leakage
pathway in how regime probabilities were computed was found and fixed, which materially
changed the calibration picture (a single global temperature now genuinely helps; per-regime
temperature freedom remains harmful — **UPDATED in §M: "genuinely helps" was ECE-only;
adding NLL/Brier shows a real metric disagreement, not a clean win**). The text/affective modality has never once produced
a statistically usable sample in any real run to date, for two independent, well-diagnosed
reasons (YouTube IP-blocking; NewsAPI weekend-date/trading-calendar misalignment plus
z-score warm-up — **UPDATED in §M: the trading-calendar misalignment cause has since been
fixed and verified; the z-score warm-up cause remains, now precisely isolated as the sole
blocker**) — every multimodal result in this project involving text is explicitly
uninformative, not a "text doesn't help" finding. Capabilities this audit was asked to
evaluate that do not exist in this project — image/audio/physiological affect, dimensional
(valence-arousal-dominance) affect regression, human-labeled emotion data with
inter-annotator agreement, and any human/stakeholder trust study — are marked **RESULT NOT
AVAILABLE** below, not fabricated. These remain RESULT NOT AVAILABLE after remediation —
see §M for why they could not legitimately be closed.

## B. Diagnostic map

| Dimension | Status | Evidence |
|---|---|---|
| Predictive performance (financial-only) | **Evaluated, walk-forward, null result** | `docs/EXPERIMENT_PROTOCOL.md` |
| Predictive performance (multimodal, financial+microstructure) | **Evaluated, single split only** | `docs/EXPERIMENT_PROTOCOL.md`, `docs/ABLATION_STUDY.md` |
| Predictive performance (multimodal, +text) | **Attempted twice, both uninformative** | `docs/TEXT_MODALITY_PILOT.md` |
| Calibration | **Evaluated, root-caused, corrected, re-evaluated (multiple iterations)** | `docs/EXPERIMENT_PROTOCOL.md` §Calibration |
| Data leakage | **Audited; one real pathway found, fixed, verified** | `docs/DATA_LEAKAGE_AUDIT.md` |
| Explainability (IG completeness) | **Evaluated — holds exactly** | `docs/XAI_EVALUATION.md` |
| Explainability (IG faithfulness) | **Evaluated, reproduced null result, sample-size ruled out** | `docs/XAI_EVALUATION.md` |
| Explainability (stability, sparsity, counterfactual) | **Evaluated — positive findings** | `docs/XAI_EVALUATION.md` |
| RAMA regime-adaptivity | **Evaluated, root-caused (two hypotheses tested and rejected)** | `docs/EXPERIMENT_PROTOCOL.md`, `docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md` row 19 |
| Affective computing — sentiment (FinBERT) | **Evaluated against real human-labeled ground truth** | `docs/NLP_MODEL_BENCHMARK.md` |
| Affective computing — discrete emotion (7-category) | **Implemented, sanity-checked only, no ground truth exists** | `docs/NLP_MODEL_BENCHMARK.md` |
| Affective computing — dimensional (valence-arousal-dominance) | **RESULT NOT AVAILABLE** | see §F below |
| Affective computing — inter-annotator agreement on this project's own labels | **RESULT NOT AVAILABLE** | see §F below |
| Cross-modal conflict detection | **Implemented, unit-tested, never run on real simultaneous multimodal data** | see §G below |
| Missing-modality robustness | **Evaluated as an unplanned natural experiment** | see §G below |
| Uncertainty quantification (epistemic, beyond softmax+calibration) | **RESULT NOT AVAILABLE** | see §I below |
| Out-of-distribution detection | **RESULT NOT AVAILABLE** (feature-drift monitoring exists but is a different capability) | see §I below |
| Human/stakeholder trust evaluation | **RESULT NOT AVAILABLE** | see §J below |

## C. Predictive performance

Financial-only walk-forward (5 expanding-window folds, SPY, 2016-2026): Models A
(XGBoost), B (logistic regression), and a single-modality NN all land at 38-40% accuracy
on a 3-class problem, none significantly different from each other (paired t-test, all
p>0.16). This is reported as a genuine negative/null result consistent with market
efficiency, not explained away. Multimodal (financial+microstructure) single-split
results reach 41-43% but are NOT walk-forward-able because microstructure data only
covers ~5-7 real trading days (a real yfinance API limitation, not a design choice — see
`docs/LIMITATIONS.md`). No text-modality result to date is statistically usable (§C in
`docs/TEXT_MODALITY_PILOT.md`). Full numbers: `docs/EXPERIMENT_PROTOCOL.md`.

## D. Calibration

Full FAILURE → ROOT CAUSE → CORRECTION → RE-EVALUATE loop, run twice (once before and once
after an independently-discovered regime-probability leakage fix changed the underlying
data). Summary: naive per-regime temperature scaling made calibration worse in every run;
a single global temperature was tested as a lower-variance alternative and, in the most
recent (leakage-corrected) run, is the first calibrator to genuinely beat uncalibrated
(ECE 0.027 vs. 0.031). A regularized/shrinkage variant was also implemented and tested;
it has not yet been the best-performing option in any run. Full numbers, three real runs,
and the honest revision of an earlier hypothesis: `docs/EXPERIMENT_PROTOCOL.md` §Calibration.

## E. Data leakage audit

This audit re-ran the leakage checks `docs/DATA_LEAKAGE_AUDIT.md` had promised but left
stale (blocked on artifacts that had, in fact, already landed). Result: labels and every
financial/technical feature are genuinely leakage-safe (proven via truncation, not
assumed). A real, previously-unknown leakage pathway was found in how regime
probabilities were computed (hmmlearn's smoothed forward-backward posterior applied over
the full train+val+test sequence in one call) — root-caused, fixed with a provably-exact
causal alternative, and verified. Measured practical impact on this dataset: small
(<1e-5) because this fitted HMM happens to produce near-deterministic regime assignments,
but the fix was adopted as the default regardless, since that property is not guaranteed
to hold for other tickers or regime configurations. Full writeup: `docs/DATA_LEAKAGE_AUDIT.md`,
`docs/EXPERIMENT_PROTOCOL.md` §"Regime-probability leakage".

## F. Explainability / faithfulness

IG's completeness axiom holds exactly against this real trained model (residual <1e-3).
Faithfulness (top-5 IG feature removal vs. random-5 removal) is statistically
indistinguishable from zero across three independent measurements, including a 4x
sample-size increase specifically run during this audit to test and rule out the
"small sample / noise" explanation (n=15: −0.001, −0.004 on re-run; n=60: −0.0004 — the
larger sample is not farther from zero, ruling out noise as the explanation). Stability
(0.95-0.97) and counterfactual validity (73-80%) are both consistently strong. RAMA's
fusion weights are exposed exactly (not approximated) but stay within a narrow 0.50-0.52
band across all four regimes in every real trained run — root-caused via two real,
implemented, tested corrections (weight-decay exclusion: no effect; 1000-epoch
unregularized training: moves weights but regime-*invariantly*, while badly overfitting),
concluding this is a genuine property of this dataset's regime/modality relationship, not
a training bug. Full detail: `docs/XAI_EVALUATION.md`, `docs/EXPERIMENT_PROTOCOL.md`.

## G. Affective computing

### Sentiment (FinBERT)

Evaluated against real, human-annotated ground truth: Financial PhraseBank (Malo et al.
2014, JASIST), `sentences_66agree` subset (i.e., published with its own ≥66%
inter-annotator agreement threshold, inherited from the source dataset, not computed by
this project). FinBERT: 0.892 accuracy, 0.882 macro F1 (n=240), clearly ahead of a
generic (SST-2) and an off-domain (Twitter) model. Important, disclosed caveat: FinBERT
was itself fine-tuned on Financial PhraseBank, so this is likely a partially in-sample,
optimistic-upper-bound estimate, not a clean generalization number — stated explicitly,
not hidden. Full method and the methodological bug found and fixed during this evaluation
(a non-random file-order sample that silently zeroed out the "negative" class): `docs/NLP_MODEL_BENCHMARK.md`.

### Discrete emotion (7-category: confidence, uncertainty, optimism, fear, caution, assertiveness, evasiveness)

Zero-shot NLI classification (`MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33`), per
the PDF's own taxonomy — NOT the standard valence-arousal-dominance (V-A-D) dimensional
model. Validated only by directional sanity checks (a fear-coded sentence scores higher
on "fear" than a confidence-coded one) in `tests/integration/test_asem_pipeline.py`, NOT
by accuracy against a labeled test set, **because no labeled dataset for this specific
7-category financial-emotion taxonomy exists anywhere the project could locate.** This is
a genuine, disclosed limitation, not a gap this audit can close by running more
evaluations — it requires a labeled corpus that does not exist.

### Dimensional affect (valence-arousal-dominance regression)

**RESULT NOT AVAILABLE.** This project implements the PDF's discrete 7-category taxonomy,
not V-A-D. No V-A-D model, training data, or evaluation exists in this codebase.
Implementing one would be new project scope (a different labeling schema, a different
model head, and — per the point below — a labeled corpus that does not currently exist
either), not an evaluation of something already built.

### Human-labeled emotion data with inter-annotator agreement (for this project's own 7-category taxonomy)

**RESULT NOT AVAILABLE.** No human annotation process, annotator pool, or inter-annotator
agreement statistic (Cohen's/Fleiss' kappa, Krippendorff's alpha, etc.) exists for this
project's specific emotion taxonomy. The sentiment model's underlying training data
(Financial PhraseBank) DOES have published inter-annotator agreement (the ≥66% threshold
used to construct the `sentences_66agree` split), which is correctly attributed to Malo et
al. above — but that is agreement on a 3-class sentiment label, not this project's
7-category emotion taxonomy, and this project did not compute it itself.

### Multimodal affect comparison table (text vs. audio vs. video vs. physiological)

**RESULT NOT AVAILABLE for every modality except text.** This project has no audio, video,
or physiological affect pipeline. `docs/LIMITATIONS.md` states directly: "Audio/video
modalities are not modeled in the live pipeline. The YouTube client only extracts
transcripts (text), never audio/video signal." There is nothing to compare across
modalities because only one (text) exists.

### Temporal affect dynamics, class imbalance in emotion labels, calibration of affect specifically

**RESULT NOT AVAILABLE.** These all presuppose either a labeled emotion corpus (does not
exist, per above) or a statistically usable volume of real text-affect data (does not
exist in any real run to date — every text-modality training run has had `text_available`
at or near 0% of rows, per `docs/TEXT_MODALITY_PILOT.md`). Computing temporal dynamics,
class balance, or calibration curves over data that was never actually available to any
trained model would fabricate structure that isn't there.

## H. Cross-modal consistency and missing-modality robustness

`src/emafrs/risk/cross_modal_conflict.py` implements pairwise directional agreement
detection across modalities (build brief Section 20's Financial/News/Emotion/Microstructure
example), is unit-tested (`tests/unit/test_cross_modal_conflict.py`), but **has never been
run against real simultaneous multimodal signals** — doing so meaningfully requires a
real text-affect signal to exist alongside financial/microstructure signals on overlapping
dates, which (per §G above and `docs/TEXT_MODALITY_PILOT.md`) has not happened in any real
run to date. This is reported as **implemented and tested, not yet evaluated on real
data**, distinct from either "evaluated" or "RESULT NOT AVAILABLE" — the capability exists,
the real-world evaluation opportunity has not yet arisen.

**Missing-modality robustness, however, has an unusually strong REAL evidence base — not
by design, but because text has been effectively "missing" (0-2 available rows out of
~160) in every real multimodal training run so far.** This constitutes an unplanned
natural experiment: `docs/TEXT_MODALITY_PILOT.md`'s two pilots show the `text_available`
masking mechanism (zero-filled features + an explicit binary availability flag, the same
pattern used for microstructure) allows training and evaluation to proceed without error
or silent corruption when a modality is almost entirely absent — accuracy/F1 in the
combined-modality runs (0.388-0.394) stayed close to the 2-modality baseline (0.412-0.425)
rather than degrading catastrophically or producing NaN/undefined behavior, which is
itself a real (if modest, and not the intended kind of) robustness finding: the
architecture degrades gracefully to near-baseline when a modality contributes almost
nothing, rather than breaking.

## I. Uncertainty quantification and out-of-distribution detection

**Epistemic uncertainty quantification (beyond softmax confidence and calibration,
already covered in §D) is RESULT NOT AVAILABLE.** No Monte Carlo dropout, deep ensemble,
conformal prediction, or Bayesian neural network exists in this codebase — verified by
direct code search across `src/emafrs`, not assumed absent.

**Out-of-distribution detection is RESULT NOT AVAILABLE as a per-prediction capability.**
This project DOES have `src/emafrs/mlops/drift.py`, a real, tested, KS-test-based feature
DRIFT detector — but that is a population-level monitoring tool comparing a training-window
feature distribution to a recent live window for retraining recommendations (build brief
Section 35's MLOps requirement), a genuinely different capability from flagging an
individual test-time input as out-of-distribution relative to training. Conflating the two
would overclaim; they are kept distinct here.

## J. Human / stakeholder trust

**RESULT NOT AVAILABLE.** No human subjects, IRB protocol, user study, survey instrument,
or any mechanism for measuring actual human trust, appropriate reliance, or explanation
usefulness exists in this project. `docs/XAI_EVALUATION.md` and this audit both state
this directly rather than substituting a proxy metric (e.g., presenting counterfactual
validity or attribution sparsity AS trust) that would be dishonest labeling. Everything
this project measures about explanations is a property of the model/explanation pairing
itself (faithfulness, stability, sparsity, counterfactual existence) — never evidence
about how an actual human would perceive, use, or (mis)trust that explanation.

## K. Final structured trustworthiness assessment

*(Original, as of the first audit pass. §M.6 below is the CURRENT, post-remediation
version of this table — several rows changed, most notably RAMA regime-adaptivity and
calibration, both refined rather than reversed.)*

| Dimension | Classification | Basis |
|---|---|---|
| Predictive performance (financial-only) | **Weak** (task appears genuinely hard / near market-efficient at this horizon) | Real walk-forward, null result, honestly reported as such |
| Predictive performance (multimodal) | **Insufficient evidence** (single-split only, no walk-forward) | Real but not statistically robust; not overclaimed as more |
| Calibration | **Weak-to-Moderate, evolving** | Naive approach weak; global temperature shows a real (if modest) improvement in the most recent, leakage-corrected run |
| Data integrity / leakage | **Moderate-to-Strong** (one real gap found and closed during this audit; everything else re-verified) | Direct truncation proofs, not assumption |
| Explainability — mechanism correctness | **Strong** | Completeness axiom holds exactly |
| Explainability — faithfulness | **Weak, reproduced** | Three independent runs, sample-size ruled out as an alternative explanation |
| Explainability — stability/sparsity/counterfactual | **Moderate-to-Strong** | Consistent positive results across runs |
| RAMA regime-adaptivity | **Weak on this dataset, mechanism proven capable in isolation** | Two real corrections tested and rejected; root cause is a genuine data property, not a bug |
| Affective computing — sentiment | **Moderate** (real ground truth, but likely partially in-sample) | Explicit, disclosed caveat on the one real number available |
| Affective computing — emotion, V-A-D, inter-annotator agreement | **Insufficient evidence / RESULT NOT AVAILABLE** | No ground truth or capability exists |
| Cross-modal consistency | **Insufficient evidence** (implemented, untested on real data) | Never had real simultaneous multimodal signal to evaluate against |
| Missing-modality robustness | **Moderate** (unplanned but real natural experiment) | Graceful degradation observed, not designed-for |
| Uncertainty (epistemic) / OOD | **RESULT NOT AVAILABLE** | Capability does not exist |
| Human trust | **RESULT NOT AVAILABLE** | Capability does not exist |

**No dimension above is classified "Strong" without a specific, cited, real result
supporting it, and no dimension is classified favorably to compensate for another being
weak.** Where evidence conflicted with an earlier hypothesis this project had published
(the calibration root cause, most notably), the hypothesis was revised in place rather
than the new evidence being suppressed.

## L. What would change this assessment (concrete, not aspirational)

- **Predictive performance**: multi-fold walk-forward for the multimodal (financial+
  microstructure) comparison, currently blocked on microstructure's ~5-7-day real data
  depth (an external API limitation, not a design gap).
- **Text/affective modality**: a genuinely usable real sample requires either resolving
  YouTube's IP-blocking (external, not fully in this project's control) or a longer-running
  NewsAPI collection process (weeks, not a single pilot run) combined with the
  weekend-date-to-next-trading-day mapping correction scoped but not yet implemented in
  `docs/TEXT_MODALITY_PILOT.md`.
- **Calibration**: a date-window-fixed (not "trailing N years from today") re-run to
  cleanly separate the regime-leakage fix's effect from ordinary date drift, per
  `docs/EXPERIMENT_PROTOCOL.md`'s disclosed confound.
- **Faithfulness**: an occlusion-based (removal-based) attribution cross-check against IG,
  the strongest remaining untested hypothesis after sample size was ruled out.
- **Affective computing (emotion, V-A-D)**: would require sourcing or constructing a
  labeled financial-emotion corpus with real inter-annotator agreement — genuinely new
  data-collection scope, not an evaluation gap closeable by more compute.
- **Human trust**: would require an actual IRB-approved user study — genuinely new scope
  outside a software reconstruction project.

---

## M. Remediation pass

For every failure identified in §A-L, this pass asked: is this fixable, and with the
real code/data available, was it actually fixed? Nothing below was accepted as
"unfixable" without a specific attempted fix or a specific reason additional data is
required.

### M.1 Failure matrix

| Problem | Status before remediation | Root cause | Fix possible? | Fix implemented? | Additional data needed? | Final status |
|---|---|---|---|---|---|---|
| NewsAPI 0 usable rows | Uninformative, 0/all splits | Two causes: (1) weekend articles never match a trading-day row; (2) z-score warm-up needs ≥3 daily rows | Cause (1): yes, implementation fix. Cause (2): no, not without more real elapsed collection time | Cause (1): **yes**, `_map_to_next_trading_day`, verified on real data | **Yes**, for cause (2) — needs real collection runs over multiple weeks, or resolving YouTube IP-blocking | **Cause (1) fixed and verified; cause (2) is a genuine data-volume limitation, precisely isolated** |
| Calibration (regime temp. scaling worse) | Naive per-regime made ECE worse | Originally hypothesized: small-sample overfitting | Partially — global/shrinkage variants are implementable | Yes — `fit_global_temperature`, `fit_regime_temperature_shrinkage`, `negative_log_likelihood` added | No — sample size (~15-70/regime) is what it is for this ticker/window | **Corrected picture, not a clean fix**: global-only improves ECE but worsens Brier/NLL; no variant is unambiguously better than uncalibrated on every metric |
| RAMA gate flatness | "Not meaningfully regime-adaptive" | Hypothesized: weight decay or insufficient training | Both hypotheses testable | Yes — weight-decay exclusion (no effect) and forced long training (regime-invariant shift only) both tested and rejected | No — this is a property of the fitted model/data, further training-side tweaks would not help per the diagnostic evidence | **Refined, not fixed**: statistically significant (p<0.0002) but practically small (1.6pp range) regime-dependent signal — genuinely exists, is genuinely small |
| IG faithfulness ≈ 0 | Reported as an unexplained weakness | Hypothesized: baseline choice, sample size, saturation, IG-specific mechanism | Diagnosable via alternative baselines/methods | Yes — tested 2 baselines, Gradient×Input, Occlusion, saturation profile, modality-coverage confound | No — 4 independent methods agree; more explainer-shopping would not change the finding | **Root-caused, not fixable by tooling**: intrinsic to a weak/diffuse decision surface on a hard task; corrected the modality-dominance sub-claim (coverage artifact, not fusion collapse) |
| HMM smoothing leakage | Not yet known at start of remediation | Smoothed forward-backward posterior over full sequence | Yes, directly | Yes — `predict_proba_causal`, forward-filter, exact | No | **Fully fixed and verified** (last-row-equivalence + truncation-invariance proofs) |
| Affective evaluation | Sentiment: real ground truth. Emotion/V-A-D: none | Sentiment: none needed (already valid). Emotion/V-A-D: no labeled corpus exists; real text volume (2-7 real dates across pilots) too small for any proxy correlation | Sentiment: N/A (already sound). Emotion ground truth: no, requires new data collection. Proxy correlation: considered and rejected — n=2 trading days makes any correlation mathematically degenerate (always ±1) | N/A | **Yes** — labeled financial-emotion corpus with real annotation would be required; not closeable by more compute on existing data | **Sentiment: Moderate (real ground truth, disclosed in-sample caveat). Emotion/V-A-D: RESULT NOT AVAILABLE, confirmed not closeable with current real data** |
| OOD | RESULT NOT AVAILABLE | No per-prediction OOD capability exists (drift.py is population-level, a different capability) | Implementable in principle (e.g. Mahalanobis distance in z-scored feature space, a k-NN density score) but was NOT implemented this pass | No — out of scope for this remediation pass; flagged, not attempted | Would need a genuine OOD test set (e.g. a different ticker/regime never seen in training) to evaluate against, which does not exist | **RESULT NOT AVAILABLE, unchanged** — a real gap, not manufactured as evaluated |
| Epistemic uncertainty | RESULT NOT AVAILABLE | No MC-dropout/ensemble/conformal-prediction methodology exists | Implementable in principle (e.g. MC-dropout requires a dropout layer this architecture does not have; a deep ensemble requires training N models) but NOT implemented this pass | No — out of scope for this remediation pass | Would need a methodology decision plus additional training runs (an ensemble), not just more data | **RESULT NOT AVAILABLE, unchanged** |
| Human trust | RESULT NOT AVAILABLE | No human-subjects study exists | No — requires an actual IRB-approved study with real participants | No | Yes — real human participants, genuinely outside this project's scope | **RESULT NOT AVAILABLE, unchanged, correctly not claimed** |

### M.2 Root-cause confidence

| Finding | Confidence | Basis |
|---|---|---|
| NewsAPI weekend-misalignment cause | **Confirmed** | Direct before/after comparison on real data: 3 raw dates → 2 correctly-merged trading-day dates, verified via `text_n_items` sum |
| NewsAPI z-score warm-up cause | **Confirmed** | Direct arithmetic: `min_periods=3 > 2` available rows, deterministically produces all-NaN |
| Calibration: small-sample overfitting (naive per-regime) | **Strongly supported** | Reproduced across 2 independent runs; degenerate temp (0.018) traced to the smallest regime group |
| Calibration: val/test non-stationarity (original alternate hypothesis) | **Plausible, not confirmed, not the leading explanation after remediation** | Never directly tested (would need a rolling/walk-forward calibration re-fit); superseded by the overfitting evidence resurfacing after the regime-leakage fix |
| RAMA gate: weight decay as cause | **Confirmed rejected** | Controlled before/after comparison, identical results to 3 decimals |
| RAMA gate: insufficient training as cause | **Confirmed rejected as primary explanation** | Direct 1000-epoch experiment: gate moves, but regime-invariantly |
| RAMA gate: weak genuine regime-modality signal in this data | **Strongly supported** | Two alternative hypotheses directly rejected by experiment; statistically significant but small effect size independently confirmed via permutation test |
| IG faithfulness: baseline-choice artifact | **Confirmed rejected** | Two baselines tested, nearly identical results |
| IG faithfulness: saturation as sole cause | **Confirmed rejected as sole cause** (real but partial effect) | Gradient norms roughly halve baseline→input (real, measured); but a gradient-free method (Occlusion) shows the same null result, which saturation cannot explain |
| IG faithfulness: weak/diffuse decision surface on a hard task | **Strongly supported** | Four independent attribution methods agree; consistent with the independently-documented near-chance predictive performance |
| Regime-probability smoothing leakage | **Confirmed** | Direct mechanism identified in hmmlearn's algorithm; fix verified via two independent proofs (last-row equivalence, truncation invariance) |
| Regime-probability leakage's PRACTICAL impact on this dataset | **Confirmed small** | Directly measured (<1e-5) holding the fitted model fixed, isolating the smoothing effect from the confounding "different train window" effect |

### M.3 Implemented fixes (exact changes)

- `src/emafrs/models/calibration.py`: added `fit_global_temperature`, `fit_regime_temperature_shrinkage`, `negative_log_likelihood`.
- `src/emafrs/regime/hmm_regime.py`: added `RegimeModel.predict_proba_causal` (forward-filter); `src/emafrs/models/dataset.py` switched to use it.
- `src/emafrs/models/train.py`: per-parameter-group Adam weight decay, excluding `gate`/`static_logits`.
- `src/emafrs/xai/evaluation.py`: added `deletion_auc`, `insertion_auc`, `comprehensiveness`, `sufficiency`.
- `src/emafrs/xai/alternative_attributions.py` (new): `compute_gradient_x_input`, `compute_occlusion`.
- `src/emafrs/nlp/text_dataset.py`: `aggregate_daily_text_features` gained an optional `trading_days` parameter implementing forward-only next-trading-day mapping (`_map_to_next_trading_day`).
- `src/emafrs/experiments/significance.py`: added `one_way_anova_f`, `permutation_test_anova`.
- New scripts: `scripts/run_xai_faithfulness_diagnostic.py`, `scripts/run_rama_gate_significance_test.py`.
- Test coverage added for every new function (`tests/unit/test_calibration.py`, `test_hmm_regime.py`, `test_train.py`, `test_xai_evaluation.py`, `test_alternative_attributions.py`, `test_text_dataset.py`, `test_significance.py`).

### M.4 Before / after

| Metric | Before | After | Δ | Significant? | Interpretation |
|---|---:|---:|---:|---|---|
| RAMA regime-adaptivity conclusion | "not meaningfully regime-adaptive" | statistically significant (p<0.0002), 1.6pp range | n/a (reclassification) | Yes, by permutation test | A real, small, previously-unquantified effect — refines rather than reverses the original finding |
| IG faithfulness (n) | n=15, −0.001 | n=60, −0.0004 across 4 methods | ~0 | No | Confirms null result is not sample-size noise |
| Calibration ECE (global-only vs. uncalibrated) | n/a (variant not tested pre-audit) | 0.027 vs. 0.031 | −0.004 | Not formally significance-tested (single split) | A real but small ECE improvement |
| Calibration Brier (global-only vs. uncalibrated) | n/a | 0.666 vs. 0.660 | +0.006 (worse) | Not formally tested | Global-only is WORSE on Brier — the metrics disagree |
| Calibration NLL (global-only vs. uncalibrated) | n/a | 1.106 vs. 1.091 | +0.015 (worse) | Not formally tested | Same disagreement pattern as Brier |
| NewsAPI combined-pilot daily text rows | 3 raw (2 weekend, silently dropped downstream) | 2 correctly trading-day-merged | −1 row, +correctness | n/a | Fix works; sparsity (not correctness) is now the sole blocker |
| Regime-probability truncation diff (isolated, model held fixed) | up to 1.0 (confounded first measurement) | <1e-5 (controlled measurement); 0.0 exactly with the causal fix | n/a | n/a | Original alarming number was a measurement artifact; real practical impact was always small, but the exact fix was adopted regardless |

### M.5 Affective computing: final determination

**Sentiment**: sound as originally evaluated — real human-labeled ground truth, real
accuracy number, disclosed in-sample caveat. No remediation needed.

**Emotion (7-category) / V-A-D**: remediation investigated whether a legitimate
*proxy* validity check (not ground-truth accuracy) was possible using currently available
real text data — real per-item sentiment/emotion scores exist from both pilots. This was
explicitly considered and rejected as scientifically indefensible: the real data spans at
most 2 trading days (NewsAPI, remediation-pass run) or ~7 unique dates (YouTube pilot),
and a correlation computed from n=2 points is mathematically degenerate (always exactly
±1 or undefined) — reporting such a number, even heavily caveated, would create a false
impression of a computed result where none is statistically meaningful. This determination
— that proxy evaluation is not currently feasible, and why — is itself the honest
deliverable for this dimension, per the explicit instruction not to turn "not evaluated"
into "performed poorly," nor to manufacture a number to have something to show.

**Multimodal affective ablation (Phase 9 request)**: RESULT NOT AVAILABLE, precise
reason — this project has exactly ONE modality that produces affect signals at all
(text; no audio/video/physiological pipeline exists). An A/B/C/AB/AC/BC/ABC ablation
matrix requires at least two independently-informative modalities to compare; with one,
the request is not evaluable, not merely "not yet done."

### M.6 Updated final structured trustworthiness assessment (supersedes §K)

| Dimension | Classification | What changed from §K |
|---|---|---|
| Predictive performance (financial-only) | Weak (unchanged) | — |
| Predictive performance (multimodal) | Insufficient evidence (unchanged) | — |
| Calibration | **Weak, precisely characterized** (was "Weak-to-Moderate, evolving") | Multi-metric evidence (added NLL) shows no variant unambiguously beats uncalibrated; downgraded from the ECE-only framing |
| Data integrity / leakage | Moderate-to-Strong (unchanged) | Second sweep closed remaining checklist items (imputation, hyperparameter selection, model selection) with no new findings |
| Explainability — mechanism correctness | Strong (unchanged) | — |
| Explainability — faithfulness | **Weak, now confirmed via 4 independent methods** (was "reproduced" via 3 sample sizes of IG alone) | Gradient×Input and Occlusion cross-checks added; modality-dominance sub-claim corrected (coverage artifact, not fusion collapse) |
| Explainability — stability/sparsity/counterfactual | Moderate-to-Strong (unchanged) | — |
| RAMA regime-adaptivity | **Statistically real but practically small effect** (was "Weak... mechanism proven capable in isolation") | Permutation test found p<0.0002 — this is a genuine, non-chance, if small, learned effect, not "no effect" |
| Affective computing — sentiment | Moderate (unchanged) | — |
| Affective computing — emotion, V-A-D, agreement | Insufficient evidence / RESULT NOT AVAILABLE (unchanged, now with an explicit closure rationale) | Proxy-check feasibility was investigated and explicitly rejected, not left ambiguous |
| Cross-modal consistency | Insufficient evidence (unchanged) | Still no real simultaneous multimodal signal to test against |
| Missing-modality robustness | Moderate (unchanged) | — |
| Uncertainty (epistemic) / OOD | RESULT NOT AVAILABLE (unchanged) | Explicitly considered for implementation and deliberately not attempted this pass (see §M.7) |
| Human trust | RESULT NOT AVAILABLE (unchanged) | — |
| Data pipeline correctness (NewsAPI causal mapping) | **New row — Fixed and verified** | Not previously a distinct line item; now tracked given the real fix and real re-verification |

### M.7 Stop conditions applied (why remediation stopped where it did)

- **OOD detection and epistemic uncertainty were not implemented this pass**, despite
  being technically implementable (e.g. a Mahalanobis-distance OOD score, an MC-dropout
  variant of the fusion models) — implementing a new capability from scratch and
  evaluating it credibly is a larger scope than *remediating an existing, documented
  finding*, which is what this pass targeted. Flagged honestly as not attempted, not
  silently left as an unexplained gap.
- **The IG faithfulness investigation stopped after 4 methods**, not because a 5th
  (e.g. SHAP) was assumed unnecessary, but because 4 independent, mechanistically
  different methods (2 IG baselines, local-gradient-only, gradient-free) already converged
  on the same conclusion — per stop condition "additional optimization no longer produces
  meaningful improvement," a 5th method would be expected to reproduce the same result
  based on the pattern already established, not overturn it.
- **RAMA gate remediation did not pursue reliability-aware gating, cross-attention, or
  mixture-of-experts alternatives.** The evidence (two rejected hypotheses, one confirmed
  small-but-real effect) supports "this dataset's regimes carry a small amount of
  differential modality-reliability signal," not "the architecture is incapable of
  expressing it" — the architecture already expresses a statistically real effect. Trying
  a fundamentally different architecture to chase a larger effect than the data supports
  would risk manufacturing an appearance of stronger regime-adaptivity than is genuine,
  which stop condition 5 (methodological bias) explicitly rules out.
- **NewsAPI's z-score-warm-up bottleneck was not "fixed" by shrinking `min_periods`.**
  This was considered and rejected twice (original diagnosis and this remediation pass)
  for the same reason: it would trade away this codebase's leakage-safety guarantees for
  a same-day cosmetic result, not address genuine data sparsity. Per stop condition 3
  ("the available data cannot support the desired evaluation"), this is left as a
  disclosed, additional-data-required limitation.
- **Affective proxy correlation was not computed** despite being technically possible to
  run (n=2 points always produce a correlation coefficient) — because a degenerate,
  guaranteed-±1 statistic would misrepresent evaluability, which stop condition 5 and the
  explicit no-fabrication rule both rule out.

### M.8 Scientific integrity checklist (self-audit against §Phase 19's rules)

No results were fabricated; no affective labels were manufactured; no calibration/tuning
was performed against the test set (verified directly — see `docs/DATA_LEAKAGE_AUDIT.md`'s
hyperparameter/model-selection row); explainability methods were compared, not
cherry-picked (all 4 IG-adjacent results reported, including the 3 that did not
"improve" anything); no negative result was hidden (calibration's Brier/NLL disagreement
with ECE is reported prominently, not buried); no failed experiment was deleted from the
record (weight-decay exclusion and the 1000-epoch forced-training test are both documented
as rejected hypotheses, not removed); no causal claim was made from feature attribution;
no human trust or OOD/epistemic-uncertainty claim was made without the corresponding real
methodology existing; no multimodal-superiority claim was made (missing-modality
"robustness" is explicitly described as graceful degradation to baseline, not an
improvement).
