# PDF → Implementation Traceability

Source document: `PROJECT DOCUMENT.pdf` — "Explainable Multimodal Financial Reasoning with
Affective Signal Fusion for Market Decision Intelligence" (AMIF / EMAFRS), filed as an Indian
patent Complete Specification (41 pages: Abstract, Field, Background, Summary of Invention,
Detailed Description §6.1–6.9, Drawings, Statement of Inventive Step, Claims 1–24).

**Important framing note.** The PDF is a *patent specification*, not a research paper or an
engineering design doc. Its mathematical formulas are illustrative/legal boilerplate ("the
mathematical formulations... are not claimed as mathematical methods... but as implemented in a
technical platform") rather than validated model equations. Several formulas as literally written
are underspecified or notationally inconsistent (e.g. Eq. 14's regime-adaptive weight uses a raw
regime index `Rt` as a real-valued input to a learned sigmoid without normalization; the RAMA
softmax in the Summary reuses `Rt` as an additive scalar shared across all modalities, which
collapses to a constant that cancels out of the softmax — as written it has no effect on relative
weights; several signals like "VPIN" and "Wasserstein Orthogonality" are named but not given
implementable definitions beyond a one-line description). Reconstruction preserves the **intent**
of every mechanism and fixes formulas that are mathematically inert as written, documenting the
fix in each case. This is disclosed explicitly rather than silently "correcting" the spec.

| # | PDF Section | Concept | Implementation Plan | Dataset / API | Experiment | Visualization |
|---|---|---|---|---|---|---|
| 1 | §6.2 | Data Acquisition Layer (DAL) | `src/data/` ingestion clients per source, common schema with `provider, timestamp, freshness, license` | yfinance, Alpha Vantage, Finnhub, NewsAPI, YouTube Data API, FRED, SEC EDGAR | Coverage report per ticker/date range | API health/freshness panel |
| 2 | §6.3 | Preprocessing / Feature Engineering | rolling z-score, forward-fill alignment, tokenization | derived from #1 | Leakage audit (no future stats in past windows) | — |
| 3 | §6.2.3 | Technical indicators (SMA/EMA/RSI/MACD/BBands) | `src/features/technical.py` using `ta`/pandas-ta on yfinance OHLCV | yfinance | Baseline model input ablation | Feature importance chart |
| 4 | §6.2.6 | High-Frequency Microstructure (OBI, LSI, OFI, VPIN, alpha decay) | **Cannot be implemented as true L2/L3 tick data** — no free API provides real order-book depth for equities/crypto at retail tier. Implemented as **documented low-frequency proxies**: OBI-proxy from bid/ask size at quote snapshot (where a provider exposes it, e.g. Alpaca/IEX for US equities) or volume-imbalance proxy from OHLCV (buy/sell volume approximation via tick rule on 1-min bars); VPIN computed on bucketed daily/minute volume, not true trade-by-trade tape | Alpha Vantage intraday, yfinance 1m bars (7-day limit), optionally Alpaca free market data | Regime-wise proxy MMIE ablation (F,G,H rows) | Manipulation-risk timeline; explicit "PROXY — not L2 data" watermark on every microstructure chart |
| 5 | §6.4 | Affective Signal Extraction Module (ASEM) | `src/nlp/asem.py`: sentiment (FinBERT), 7-class emotion head (fine-tuned), role stratification from metadata | NewsAPI, YouTube transcripts, SEC EDGAR filings text | Transformer benchmark table (§Model Comparison) | SHAP/IG token attribution, hype timeline |
| 6 | §6.4.1 | Influencer role stratification (5 roles) | Role assigned only when metadata supports it (channel/source category, SEC filer role); unknown → `role=unclassified`, never inferred from name | NewsAPI source field, YouTube channel metadata, SEC EDGAR filer type | Role-stratified sentiment reliability | Role breakdown bar chart |
| 7 | §6.5 | Market Regime Detection Engine (MRDE) — HMM over volatility, entropy, Hurst | `src/regime/hmm_regime.py` — `hmmlearn` GaussianHMM, 4 states, features = realized vol, Shannon entropy, Hurst (R/S), volume z-score | yfinance OHLCV | Regime quality eval (state persistence, silhouette vs. VIX-like proxy) | Regime timeline overlay on price |
| 8 | §6.6 | Market Microstructure Intelligence Engine (MMIE) | `src/microstructure/mmie.py`, consumes proxies from #4 | see #4 | Ablation F/G rows | Microstructure panel |
| 9 | §6.7 | Volatility-Aware Adaptive Fusion Engine (VAFE) / RAMA | `src/models/rama.py` — learned gated fusion; regime enters as **one-hot embedding**, not raw scalar (fixes the inert-softmax issue in #Framing note above), documented deviation | derived features | Model D/E vs. Model C (static fusion) ablation — primary research question | RAMA weight-over-time chart |
| 10 | §6.7.3 | Volatility-aware calibrated inference (temperature scaling by regime) | `src/models/calibration.py` — regime-conditioned temperature scaling fit post-hoc on validation set | — | ECE / Brier per regime | Reliability diagram per regime |
| 11 | §6.8 | Explainability Module — Integrated Gradients, attention, NLG | `src/xai/` — Captum IntegratedGradients, attention rollout, template NLG | — | Faithfulness, stability, sparsity metrics (§21 of build brief) | SHAP summary, IG waterfall |
| 12 | §6.8.1 / Claims 1,5,14,24 | Counterfactual explainability (`Decision(M,S,R)` vs `Decision(M,S+Δ,R)`) | `src/xai/counterfactual.py` — feature-ablation and perturbation counterfactuals over the actual trained model, minimum-Δ search | — | Counterfactual validity rate | Counterfactual bar chart |
| 13 | §6.6.5, Claim 12 | Extended 5-signal manipulation risk model | `src/risk/manipulation_risk.py` — learned weights `w1..w5`, regime-modulated | derived | Precision/recall vs. labeled historical pump-and-dump cases (SEC litigation releases as weak labels) | Manipulation risk timeline |
| 14 | §6.9, Claim 15 | Execution-aware alert / order-management interface | Out of scope for real order routing (no live brokerage integration); implemented as a **simulated recommendation output** (size reduction / venue routing suggestion), clearly labeled non-executing | — | — | Alert panel |
| 15 | §6.9, Claim 16-17 | Citizen alert delivery + Global Adaptive Integrity Controller | `src/alerts/` — in-app/dashboard notification + optional email/webhook; integrity controller = scheduled job monitoring live precision/FPR and flagging threshold drift | — | Drift-detection test | Alert feed + controller status panel |
| 16 | Fig. 1–4 | Architecture diagrams | Reproduced as Mermaid diagrams in `docs/SYSTEM_ARCHITECTURE.md`, not the patent's line-art | — | — | Architecture diagram |
| 17 | §9, Claims | Legal novelty/claims language | Not implemented (legal argument, not a system requirement); retained only as intent context | — | — | — |
| 18 | §6.2.4 | AMSF/FTCS/Signal Confidence/Adaptive Signal Weighting (ω_MACD etc.) | **Not implemented.** Unlike OBI/RSI/MACD/CVaR (all given explicit, standard formulas), the PDF names these as outputs the DAL "ingests" from unspecified "ancillary algorithmic scoring systems" — no formula is given anywhere in the specification. Inventing a formula and presenting it as "the system's AMSF score" would be fabrication, not reconstruction; documented as unavailable rather than invented. | — | — | — |
| 19 | §6.7, Summary §4 | RAMA / VAFE fusion (Models C vs D) | `src/emafrs/models/fusion.py` (StaticFusionModel=C, RAMAFusionModel=D), `models/train.py`. RAMA's mechanism is proven regime-dependent in unit tests (`test_rama_weights_differ_across_regimes`), but every real training run to date on SPY (financial+microstructure) learns a gate that stays close to uniform (financial weight 0.502-0.518 across all 4 observed regimes, reproduced across 2 independently-trained runs). **Root-caused, not just observed** (`docs/EXPERIMENT_PROTOCOL.md`'s Fusion-gate diagnostic): (1) excluding the gate parameter from Adam weight decay was tested and made no measurable difference, ruling out optimizer regularization as the cause; (2) forcing 1000 epochs of unregularized training (no early stopping, well past the point of overfitting — train loss fell to 0.096) DID move the gate away from uniform, but to a roughly regime-INVARIANT global preference for the financial modality (0.85-0.90 across all regimes, not per-regime differentiated) — meaning even maximal gradient pressure does not produce genuine regime-specific specialization on this data. Conclusion: this real dataset's HMM-detected volatility regimes do not carry strong differential information about which modality (financial vs. microstructure) is more reliable, so a model trained with principled early stopping correctly declines to invent regime-specific preferences the data does not support — RAMA ≈ static fusion here is a property of this data/regime-detection combination, not a training bug or an optimizer artifact. Reported honestly, not hidden. | yfinance | `scripts/run_baseline_comparison.py`, `scripts/run_regime_calibration_xai_eval.py` | Cross-ticker/cross-window replication to test whether SOME ticker/regime combination shows genuine differentiation (not yet run) |

## Formula corrections made explicit

1. **RAMA softmax (Summary §4).** As literally written, `Rt` is a scalar added identically inside
   every modality's exponent, which is constant across `i` and cancels in the softmax — the
   equation as printed cannot make attention regime-dependent. Reconstruction implements the
   clear intent (regime-conditioned attention) via a regime-embedding term `γ_i(R_t)` that is
   **modality-specific**, i.e. `α_i(R_t) = softmax_i(I_i - λL_i + W_γ[R_t]_i)`, where `W_γ` is a
   learned `(num_regimes × num_modalities)` matrix. This is the smallest change that makes the
   mechanism actually regime-adaptive, and is documented as a deviation, not silently applied.
2. **VPIN (§6.6.3).** Textbook VPIN requires trade-classified tick data bucketed by volume; free
   data sources only provide bar aggregates. Implemented as a bar-level approximation using the
   Lee-Ready tick rule on 1-minute bars where available, degrading to a volume-imbalance proxy
   otherwise, with the approximation stated on every output that surfaces it.
3. **Hurst Exponent (Eq. 2).** Implemented via classical rescaled-range (R/S) analysis as
   specified; documented as a noisy estimator at the short (≤252-day) windows realistically
   available and cross-checked against a DFA (detrended fluctuation analysis) estimator as a
   sensitivity check, per the "evaluate regime quality" requirement.

## What cannot be implemented at all (documented, not silently dropped)

- True Level-2/Level-3 order book snapshots at tick/microsecond resolution — no free/research-tier
  API grants this for equities or major crypto pairs. All "microstructure" outputs in this
  reconstruction are labeled **proxy** and their limitations are in `docs/LIMITATIONS.md`.
- Real audio/video affect modeling from YouTube content beyond transcripts, unless a video is
  actually downloaded and processed — feasible only for a small curated case-study set given
  compute/storage limits, not the full live pipeline. Reported as "textual modality derived from
  audiovisual source" per the build brief's own instruction when this applies.
- Execution-aware order routing into a live brokerage/OMS — no such integration exists in this
  reconstruction; outputs are advisory/simulated only.
