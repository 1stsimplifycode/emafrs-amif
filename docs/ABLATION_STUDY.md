# Ablation Study

Build brief Section 23 specifies an 11-row ablation grid (A-K). This document states,
row by row, which are evaluated with real results, and which are honestly marked
**Not yet evaluated** with the specific reason — never silently omitted, never filled
with an invented number.

| Row | Combination | Status | Result |
|---|---|---|---|
| A | Financial only | **Evaluated** | Model B (logistic regression, financial-only): walk-forward mean accuracy 0.390, 95% CI [0.374, 0.406], 5 folds. Model A (XGBoost, financial-only): 0.384, CI [0.349, 0.419]. See `docs/EXPERIMENT_PROTOCOL.md`. |
| B | Text only | **Attempted; underpowered, not conclusive** | NewsAPI/YouTube keys were added and YouTube now works. A real pilot (`docs/TEXT_MODALITY_PILOT.md`, `scripts/fetch_youtube_text_dataset.py`) searched real videos and processed real transcripts through ASEM, but only 2 of ~160 test rows ended up with `text_available=1` (YouTube IP-blocked 23/35 transcript requests, and the rolling z-score warm-up consumed most of the remaining sparse signal). Text-only (dropping financial/microstructure entirely) was not separately run, since 2 usable rows cannot support training any model. |
| C | Financial + Text | **Attempted; underpowered, not conclusive** | Same pilot as row B, jointly with financial+microstructure (see row H) rather than run in isolation. Result statistically indistinguishable from the 2-modality baseline, and NOT interpreted as "text doesn't help" -- the sample was too small to conclude anything. See `docs/TEXT_MODALITY_PILOT.md`. |
| D | Financial + Emotion | **Not yet evaluated in isolation** | The pilot's text features already include the 7-category emotion vector alongside sentiment/hype (they were fetched together, not separable post-hoc without re-running ASEM on a financial-plus-emotion-only feature subset). Not run as a distinct combination. |
| E | Financial + Text + Emotion | **Attempted; underpowered, not conclusive** | Emotion features ARE included in the pilot's text modality (`text_emotion_*` columns) -- see row H's result, which is this combination plus microstructure. |
| F | Financial + Microstructure | **Evaluated (single split only)** | RAMA (financial+microstructure) single-split test accuracy 0.425, macro F1 0.259 (`docs/EXPERIMENT_PROTOCOL.md`). NOT walk-forward-able: microstructure only covers the trailing ~5-7 real trading days (yfinance's 1-minute-bar cap), so this row cannot currently be repeated across multiple historical folds without fabricating "as of" microstructure availability for dates that never had real intraday data. |
| G | Financial + Text + Microstructure | **Attempted -- see row H** | This is exactly the 3-modality pilot combination; see row H for the result and its interpretation. |
| H | Full multimodal (financial+text+emotion+microstructure) | **Attempted three times; all uninformative, one root cause now fixed** | YouTube-only pilot: Static accuracy 0.394/macro F1 0.263, RAMA accuracy 0.388/macro F1 0.252, with 2/~160 test rows carrying real text signal. Combined YouTube+NewsAPI pilot: same-looking numbers but **0/all** rows carrying real text signal -- a degenerate run where the text feature block was a constant zero vector. Remediation-pass re-run, after fixing the weekend-date/trading-calendar misalignment (articles now map forward-only to the next trading day, verified correct): still **0/all** rows available, because a second, independent cause (z-score warm-up needing >=3 total daily text rows; NewsAPI's ~29-day lookback cap produced only 2) is now the sole remaining, precisely-characterized blocker -- a genuine additional-data-required limitation, not a further implementation bug. None of the three runs supports any claim about text modality value. Full details and root-cause diagnosis are in `docs/TEXT_MODALITY_PILOT.md`. |
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

Of the build brief's 11-row primary grid: **2 rows have real single-split results (F, I)**,
**1 row has both single-split and (degenerate, financial-only) walk-forward results (J)**,
**1 row has walk-forward-only results because it needs no additional modality (A)**, and
**6 rows involving text (B, C, D, E, G, H) now have a real attempt** (once YouTube API
access was added) rather than being blocked on missing keys, but the result is
**underpowered and inconclusive**, not a null result -- YouTube's undocumented transcript
IP-blocking (23/35 requests blocked in the real run) plus the rolling z-score warm-up
consuming most of the remaining sparse signal left only 2 of ~160 test rows with real text
data. A follow-up run adding NewsAPI as a second real source (`docs/TEXT_MODALITY_PILOT.md`,
"Follow-up" section) made this *worse*, not better: 0/all rows had real text signal, due to
two compounding, precisely diagnosed causes (weekend-published articles never matching a
trading-day row; the same z-score warm-up problem). Full details in
`docs/TEXT_MODALITY_PILOT.md`. This is honestly reported as "not
enough data to conclude anything," never rounded up to "text doesn't help" or down to
"text wasn't tried." Row K is answered by direct XAI evaluation
(`docs/XAI_EVALUATION.md`) rather than a predictive-metric ablation, since that is what the
build brief's own question ("does XAI add value") actually requires.
