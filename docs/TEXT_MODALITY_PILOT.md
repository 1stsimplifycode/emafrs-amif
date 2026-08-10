# Text Modality Pilot: First Real Run

This documents the first real (not synthetic, not zero-filled-everywhere) attempt to
populate the text/affective modality with actual YouTube-derived data and train a
3-modality (financial + microstructure + text) RAMA/static fusion model on it. It
directly follows up on the "Not yet evaluated" rows in `docs/ABLATION_STUDY.md`.

Reproduce: `python scripts/fetch_youtube_text_dataset.py`. Pipeline code:
`src/emafrs/nlp/text_dataset.py` (search -> transcript -> ASEM -> daily aggregation),
wired into `emafrs.models.dataset.build_daily_dataset_from_ohlcv` via a `text_daily`
parameter, following the exact same disclosed-masking pattern already used for the
microstructure modality (`text_available` column, zero-filled feature columns on days
with no video, never fabricated).

## What was run

- Query: `"Federal Reserve interest rate"` (a market-wide, SPY-relevant communication
  source, matching the ticker already used across this project's experiments).
- Window: 2026-02-09 to 2026-08-10 (~6 months), chunked into 7 30-day windows.
- YouTube quota used: 700 units (7 `search.list` calls @ 100 units each) — well within
  the 10,000/day budget.

## Real results

| | |
|---|---|
| Videos found | 35 |
| Videos with a retrievable transcript | 12 (34%) |
| Videos successfully processed by ASEM | 12 |
| Unique calendar dates with text data | 7 |
| Test-set rows with `text_available=1` | **2** (out of ~160 test rows) |

Real per-video output (FinBERT sentiment, ASEM hype score) — genuine model output on
genuine transcript text, not illustrative:

| Published | Sentiment | Hype | Title |
|---|---:|---:|---|
| 2026-03-10 | −0.879 | 0.000 | How the Fed's decision on interest rates could affect inflation |
| 2026-02-09 | −0.413 | 0.106 | Why the Federal Reserve Raises Interest Rates \| Economics Explained |
| 2026-03-13 | −0.180 | 0.092 | The Fed Meeting That Could Break the Market (March 2026) |
| 2026-03-18 | +0.003 | 0.007 | BREAKING: Federal Reserve CANCELS Rate Cuts... |
| 2026-04-29 | −0.016 | 0.000 | Federal Reserve holds interest rates steady |

### Training result (3-modality, single split)

| Model | Accuracy | Macro F1 | AUC | Mean fusion weights |
|---|---:|---:|---:|---|
| Static | 0.394 | 0.263 | 0.521 | financial 0.349, microstructure 0.322, text 0.329 |
| RAMA | 0.388 | 0.252 | 0.525 | financial 0.339, microstructure 0.331, text 0.330 |

## Honest interpretation: this does NOT answer the ablation question yet

These numbers are statistically indistinguishable from the earlier 2-modality
(financial+microstructure) single-split results in `docs/EXPERIMENT_PROTOCOL.md`
(accuracy 0.412/0.425). **This is expected and uninformative, not a finding that "text
doesn't help"** — with only 2 of ~160 test rows actually carrying real text signal, no
model could learn anything from the text modality in this run, and no conclusion about
its value can be drawn. Reporting this as "text was tested and found not to help" would
be a fabricated conclusion from an underpowered sample; this document explicitly avoids
that framing.

## Two real, disclosed limitations found by actually running this

1. **YouTube transcript IP-blocking**: 23 of 35 found videos failed with `"YouTube is
   blocking requests from your IP"` — an undocumented anti-bot heuristic in the
   `youtube-transcript-api` library's underlying (unofficial) endpoint, not a bug in this
   project's code or a documented, fixed rate limit. A 1.5-second courtesy delay between
   transcript requests was added (`fetch_video_affective_records`'s
   `transcript_request_delay_seconds` parameter) to reduce (not eliminate) this. A
   dedicated lightweight integration test
   (`tests/integration/test_youtube_text_dataset.py`) confirmed the SAME blocking
   recurred on a much smaller, single-video attempt made shortly after — this appears to
   be a sustained IP-level restriction from this development environment's network, not
   an artifact of request volume alone. The test skips gracefully when this happens
   rather than failing, since it is a real external condition to report, not a defect to
   hide behind a passing test.
2. **Rolling z-score warm-up consumes most of the sparse real signal**: with only 7
   unique text-bearing dates spread across ~6 months, `text_zscore_window =
   min(21, max(3, len(text_daily)-1)) = 6` means the first 5 of 7 rows are NaN
   (insufficient window), leaving only 2 usable rows. This is the SAME leakage-safe
   design already used for every other rolling feature in this codebase (no shortcuts
   were taken for text), but it interacts badly with genuinely sparse real-world video
   coverage. A larger date range (to accumulate more unique dates) and/or a smaller
   minimum window specifically tuned for text sparsity are the natural next steps, not
   yet done here.

## What this pilot proves and doesn't prove

**Proves**: the full pipeline (real YouTube search → real transcript fetch → real ASEM
processing → real daily aggregation → real dataset join with proper masking → real 3-
modality model training) runs end-to-end without errors on genuinely live data, and
produces real, inspectable per-video model output.

**Does not prove**: anything about whether the text modality improves prediction quality
— the sample obtained was too small and too blocked-by-YouTube to test that. See
`docs/ABLATION_STUDY.md` for the updated status of the text-dependent ablation rows.

## Next steps (not yet done)

- Run over a longer historical window (a year or more) to accumulate more unique
  text-bearing dates, budgeting YouTube quota accordingly (100 units per 30-day window).
- Consider multiple queries (not just "Federal Reserve interest rate") to increase video
  yield per window.
- Investigate whether the IP-blocking is specific to this network/environment and
  whether it clears with time or persists.
- Once enough usable rows accumulate, re-run the walk-forward/ablation comparisons with
  text genuinely included, with a real (not underpowered) sample.
