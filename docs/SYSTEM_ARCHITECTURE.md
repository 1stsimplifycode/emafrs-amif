# System Architecture

Reconstructs the PDF's modular pipeline (Section 6.1, Figures 1-4) as a real, tested
Python package (`src/emafrs/`). Modularity is deliberate and preserved: each layer below
is a separate package with its own tests, not collapsed into one end-to-end model.

## Pipeline overview

```mermaid
flowchart TD
    A[Data Acquisition Layer<br/>emafrs.data] --> B[Preprocessing / Feature Engineering<br/>emafrs.features]
    A --> C[Affective Signal Extraction Module<br/>emafrs.nlp]
    B --> D[Market Regime Detection Engine<br/>emafrs.regime]
    A --> E[Market Microstructure Intelligence Engine<br/>emafrs.microstructure]
    D --> F[Regime-Adaptive Multimodal Attention /<br/>Volatility-Aware Fusion Engine<br/>emafrs.models.fusion]
    B --> F
    E --> F
    C -.text modality, not yet populated.-> F
    F --> G[Manipulation Risk Model<br/>emafrs.risk]
    C --> G
    E --> G
    D --> G
    F --> H[Explainability Layer<br/>emafrs.xai]
    H --> I[Decision Intelligence Output /<br/>Citizen Alerts]
    G --> I
```

## RAMA mechanism (the core inventive mechanism, PDF Summary "First Inventive Mechanism")

```mermaid
flowchart LR
    subgraph Modalities
        M1[Financial features]
        M2[Microstructure proxies]
        M3["Text/emotion (not yet populated)"]
    end
    subgraph Encoders
        E1[ModalityEncoder]
        E2[ModalityEncoder]
        E3[ModalityEncoder]
    end
    R[Regime posterior<br/>n_regimes-dim] --> G["Learned gate matrix<br/>(n_regimes x n_modalities)"]
    M1 --> E1 --> W
    M2 --> E2 --> W
    M3 --> E3 --> W
    G --> SM[Softmax over modalities]
    SM --> W[Weighted fusion]
    W --> H[Classification head]
    H --> P["P(bearish/neutral/bullish)"]
```

Note: this is the DISCLOSED-corrected version of the PDF's RAMA formula — see
`docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md` row #19 and `src/emafrs/models/fusion.py`
docstring for why the literal patent formula could not be implemented as printed.

## Module map

| PDF concept | Package | Key files |
|---|---|---|
| Data Acquisition Layer | `emafrs.data` | `yfinance_client.py`, `alphavantage_client.py`, `finnhub_client.py`, `newsapi_client.py`, `youtube_client.py`, `fred_client.py`, `sec_edgar_client.py`, `health.py` |
| Preprocessing / Feature Engineering | `emafrs.features` | `technical.py`, `preprocessing.py`, `risk_metrics.py`, `leakage_audit.py` |
| Affective Signal Extraction Module (ASEM) | `emafrs.nlp` | `sentiment.py`, `emotion.py`, `semantic_encoder.py`, `roles.py`, `credibility.py`, `hype.py`, `asem.py`, `sentiment_benchmark.py` |
| Market Regime Detection Engine (MRDE) | `emafrs.regime` | `info_dynamics.py` (Hurst/entropy), `hmm_regime.py`, `evaluation.py` |
| Market Microstructure Intelligence Engine (MMIE) | `emafrs.microstructure` | `proxies.py` |
| Volatility-Aware Adaptive Fusion Engine / RAMA | `emafrs.models` | `dataset.py`, `labels.py`, `baselines.py`, `fusion.py`, `train.py`, `calibration.py` |
| Manipulation risk model, cross-modal conflict | `emafrs.risk` | `manipulation_risk.py`, `cross_modal_conflict.py` |
| Explainability layer | `emafrs.xai` | `integrated_gradients.py`, `counterfactual.py`, `nlg.py`, `evaluation.py` |
| Experiment protocol | `emafrs.experiments` | `walk_forward.py`, `significance.py` |
| Helper dev tooling (not a research claim) | `emafrs.mcp_server` | `tools_data.py`, `tools_inspect.py`, `server.py` |

## Data flow and provenance

Every record produced by `emafrs.data` carries `(provider, endpoint, fetched_at,
observation_time, license_note, cache_hit)` (`emafrs.data.schema.SourceRecord`) so nothing
downstream can lose track of where a number came from or how fresh it is. See
`docs/DATA_SOURCES_AND_APIS.md`.

## What is NOT collapsed into a single model

The build brief explicitly warns against collapsing the architecture into one generic
neural network. Concretely, in this implementation:
- Regime detection (`fit_regime_hmm`) is a **separate, inspectable HMM**, not a hidden
  layer inside the fusion network — its transition matrix, per-regime feature means, and
  quality report (persistence, separation) are all directly queryable.
- The manipulation risk score is a **separate, disclosed weighted combination**
  (`compute_manipulation_risk`) over the fusion model's inputs, not a learned head buried
  inside the neural network — its weights are inspectable and, by default, an equal-weight
  prior rather than an opaque fit.
- The explainability layer operates on the **already-trained, frozen** fusion model —
  it is a genuinely separate analysis step (Integrated Gradients, counterfactual search),
  not something the model optimizes for during training.

## Production architecture (planned, not yet built)

FastAPI backend + persistent DB + dashboard + Docker are tracked separately — see
`docs/PRODUCTION_DEPLOYMENT.md` and `docs/MLOPS.md` once that phase lands. This document
describes the research/modeling architecture that exists today.
