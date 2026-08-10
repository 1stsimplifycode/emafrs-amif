# Recovery and Reconstruction

## What happened

The original implementation of the system described in `PROJECT DOCUMENT.pdf` (an Indian
patent Complete Specification titled *"Explainable Multimodal Financial Reasoning with
Affective Signal Fusion for Market Decision Intelligence,"* internally referred to as
EMAFRS / AMIF) was lost to a system crash with no backups. When this reconstruction began,
`C:\Users\mailt\Downloads\CAPSTONE` contained exactly one file: the PDF itself.

## What this reconstruction is, and is not

This is a **from-scratch reimplementation**, not a recovery of any surviving code. No file
in this repository was copied, decompiled, or otherwise derived from a prior codebase --
none existed to derive from. Every module was designed and written against the PDF as the
authoritative specification of *intent*, per the brief's framing of the PDF as a "contract
of architectural continuity": the system had to remain recognizably the one described in
the patent, while implementation details, model choices, datasets, and engineering
practices were free (and required) to be modernized and made rigorous.

## Method

1. **Read the full specification first** (41 pages) before writing any code. Extracted
   every named module, mathematical formulation, and claim into
   `docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md` before starting implementation.
2. **Flagged what the spec's formulas actually say versus what they were clearly intended
   to do.** One example: the RAMA (Regime-Adaptive Multimodal Attention) softmax as
   literally printed adds a scalar regime index identically to every modality's logit,
   which cancels out algebraically and cannot make attention regime-dependent. This was
   caught by reading the formula carefully, not just implementing it uncritically, and a
   disclosed, minimal correction was applied (modality-specific regime gate matrix) --
   see traceability doc row #19 and `src/emafrs/models/fusion.py`.
3. **Distinguished "not specified" from "not available."** Several PDF-named signals
   (AMSF, FTCS, Adaptive Signal Weighting parameters) are described as ingested from
   unspecified "ancillary algorithmic scoring systems" with no formula given anywhere in
   the document. Rather than inventing formulas and presenting them as the patent's own
   system, these are documented as deliberately unimplemented (traceability doc row #18).
4. **Built module-by-module in dependency order**, validating each against real external
   data (live yfinance/SEC EDGAR/NewsAPI calls, real Financial PhraseBank data, a real
   trained model) before moving to the next, rather than building the full pipeline
   against mocked data and hoping the pieces would compose correctly later. Every phase's
   git commit records what was validated and how.
5. **Reported negative and inconvenient findings as prominently as positive ones.**
   Several real results in this project are not flattering (RAMA's fusion weights did not
   turn out to be strongly regime-differentiated in the trained runs; temperature-scaling
   calibration got worse, not better, on one test split; IG faithfulness came out near
   zero for the trained model). These are documented in
   `docs/EXPERIMENT_PROTOCOL.md` and `docs/XAI_EVALUATION.md` exactly as they were
   measured, because the project's standing instruction is scientific honesty over a clean
   narrative.

## What could not be reconstructed exactly

See `docs/LIMITATIONS.md` for the full list; the two structural ones are:
- **True Level-2/Level-3 order book data.** No free/research-tier API available to this
  project provides real bid/ask depth or trade-by-trade tick data. The Market
  Microstructure Intelligence Engine is built on documented OHLCV-bar proxies instead,
  labeled as such everywhere they appear.
- **Full audio/video modeling of communications.** The YouTube pipeline extracts video
  metadata and transcript TEXT only; it does not download or process audio/video streams
  for affect features at pipeline scale, consistent with the build brief's own instruction
  to report this honestly as "textual modality derived from an audiovisual source."

## Provenance of this document

This file itself is the product of the same reconstruction effort it describes -- it was
written after all ten core research phases (data acquisition through experiment protocol)
were complete and validated, not drafted speculatively in advance of the work.
