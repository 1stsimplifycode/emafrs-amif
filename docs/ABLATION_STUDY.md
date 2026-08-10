# Ablation Study

Build brief Section 23 specifies an 11-row ablation grid (A-K). This document states,
row by row, which are evaluated with real results, and which are honestly marked
**Not yet evaluated** with the specific reason — never silently omitted, never filled
with an invented number.

| Row | Combination | Status | Result |
|---|---|---|---|
| A | Financial only | **Evaluated** | Model B (logistic regression, financial-only): walk-forward mean accuracy 0.390, 95% CI [0.374, 0.406], 5 folds. Model A (XGBoost, financial-only): 0.384, CI [0.349, 0.419]. See `docs/EXPERIMENT_PROTOCOL.md`. |
| B | Text only | **Not yet evaluated** | No NewsAPI/YouTube key configured in this environment; ASEM produces no real text features at dataset-build time (see `docs/LIMITATIONS.md`). Architecturally supported (`emafrs.nlp.asem`) and validated in isolation (`tests/integration/test_asem_pipeline.py`) but not wired into a trainable dataset. |
| C | Financial + Text | **Not yet evaluated** | Same blocker as row B. |
| D | Financial + Emotion | **Not yet evaluated** | Same blocker as row B (emotion is a component of the text/ASEM pipeline). |
| E | Financial + Text + Emotion | **Not yet evaluated** | Same blocker as row B. |
| F | Financial + Microstructure | **Evaluated (single split only)** | RAMA (financial+microstructure) single-split test accuracy 0.425, macro F1 0.259 (`docs/EXPERIMENT_PROTOCOL.md`). NOT walk-forward-able: microstructure only covers the trailing ~5-7 real trading days (yfinance's 1-minute-bar cap), so this row cannot currently be repeated across multiple historical folds without fabricating "as of" microstructure availability for dates that never had real intraday data. |
| G | Financial + Text + Microstructure | **Not yet evaluated** | Blocked on both row B/C's and row F's limitations simultaneously. |
| H | Full multimodal (financial+text+emotion+microstructure) | **Not yet evaluated** | Blocked on text availability; would additionally inherit row F's walk-forward limitation. |
| I | Static multimodal fusion | **Evaluated (single split only)** | Model C (`StaticFusionModel`, financial+microstructure): see `scripts/run_baseline_comparison.py` output — accuracy 0.412 on the same single split. |
| J | RAMA | **Evaluated (single split + partial walk-forward)** | Single-split (financial+microstructure): accuracy 0.425 (row F). Walk-forward (financial-only, where RAMA degenerates to a single-modality fusion network since there is only one modality to gate): mean accuracy 0.402, CI [0.379, 0.426], 5 folds — not statistically distinguishable from Models A/B (`docs/EXPERIMENT_PROTOCOL.md`). |
| K | RAMA + XAI | **Evaluated for integration, not as a distinct ablation** | The XAI layer (Integrated Gradients, counterfactual, NLG) is proven to run end-to-end against the trained RAMA model (`tests/integration/test_xai_end_to_end.py`) and quantitatively evaluated (`docs/XAI_EVALUATION.md`). Per `docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md`, Model E is architecturally identical to Model D — XAI is a read-only layer over the same trained weights, so it cannot itself change accuracy/F1; there is no distinct "Model E accuracy" to report. What IS evaluated is whether the XAI layer's explanations are faithful/stable/sparse (see XAI_EVALUATION.md) — i.e., whether adding XAI produces USEFUL evidence, which is the build brief's actual question ("does XAI add explanatory value") rather than a predictive-accuracy ablation. |

## Modality-removal and mechanism-removal ablations (build brief Section 23, second list)

| Ablation | Status | Result |
|---|---|---|
| Remove each modality individually | **Partially evaluated** | Financial-only vs. financial+microstructure compared in the single split (rows A, F above). Text/emotion removal not evaluable (never included to begin with — see row B). |
| Replace RAMA with static fusion | **Evaluated** | Model C (static) vs. Model D (RAMA), single split, financial+microstructure: static accuracy 0.412 vs. RAMA accuracy 0.425 — a 1.3-point difference on ONE split, not tested for significance (would need multiple seeds/folds on this modality combination, which walk-forward cannot currently provide — see Scope limitation in `docs/EXPERIMENT_PROTOCOL.md`). Not claimed as a proven RAMA advantage. |
| Remove regime conditioning | **Evaluated (proxy)** | The RAMA-vs-static comparison above IS this ablation: StaticFusionModel has no regime input at all, RAMAFusionModel does. Same caveat: single split, not significance-tested. |
| Remove XAI | **N/A** | XAI is a post-hoc, read-only layer (Integrated Gradients/counterfactual computed from a frozen trained model) — it cannot be "removed" in a way that changes predictive metrics, since it never participates in training or inference decisions. "Removing" it just means not computing explanations, which has no ablation-relevant effect to measure. |
| Remove counterfactual analysis | **N/A**, same reasoning as above. | Counterfactual search is diagnostic-only over a frozen model. |

## Honest summary

Of the build brief's 11-row primary grid, **2 rows have real single-split results (F, I)**,
**1 row has both single-split and (degenerate, financial-only) walk-forward results (J)**,
**1 row has walk-forward-only results because it needs no additional modality (A)**, and
**6 rows are blocked on the text modality not being populated** (B, C, D, E, G, H), which
requires a configured NewsAPI/YouTube key per `docs/LIMITATIONS.md` — not a modeling
limitation, a data-availability one. Row K is answered by direct XAI evaluation
(`docs/XAI_EVALUATION.md`) rather than a predictive-metric ablation, since that is what the
build brief's own question ("does XAI add value") actually requires.
