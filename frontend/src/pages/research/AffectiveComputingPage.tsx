import "../../styles/research.css";

export function AffectiveComputingPage() {
  return (
    <div>
      <div className="research-page-head">
        <span className="eyebrow">Affective Computing</span>
        <h1 className="display-xl">What EMAFRS's text/sentiment component can and can't claim</h1>
      </div>

      <div className="card finding-card">
        <div className="finding-card-head">
          <span className="display-md">Sentiment (FinBERT)</span>
          <span className="verdict-badge verdict-fixed">Validated against real ground truth</span>
        </div>
        <div className="metric-grid">
          <div className="metric-tile">
            <span className="metric-tile-label">Accuracy</span>
            <span className="metric-tile-value">0.892</span>
          </div>
          <div className="metric-tile">
            <span className="metric-tile-label">Macro F1</span>
            <span className="metric-tile-value">0.882</span>
          </div>
          <div className="metric-tile">
            <span className="metric-tile-label">n</span>
            <span className="metric-tile-value">240</span>
          </div>
        </div>
        <p>
          Evaluated against Financial PhraseBank (Malo et al. 2014), a real human-labeled
          dataset with its own published ≥66% inter-annotator agreement threshold.
          Disclosed caveat: FinBERT was itself fine-tuned on this dataset, so this is
          likely a partially in-sample, optimistic-upper-bound estimate, not a clean
          generalization number.
        </p>
      </div>

      <div className="card finding-card">
        <div className="finding-card-head">
          <span className="display-md">Discrete emotion (7-category taxonomy)</span>
          <span className="verdict-badge verdict-mixed">No ground truth exists</span>
        </div>
        <p>
          Zero-shot NLI classification across confidence, uncertainty, optimism, fear,
          caution, assertiveness, and evasiveness — the PDF's own taxonomy, not
          valence-arousal-dominance. Validated only by directional sanity checks (a
          fear-coded sentence scores higher on "fear" than a confidence-coded one), never
          by accuracy against a labeled set, because no labeled dataset for this specific
          taxonomy exists anywhere the project could locate.
        </p>
      </div>

      <div className="card finding-card">
        <div className="finding-card-head">
          <span className="display-md">Dimensional affect (V-A-D), inter-annotator agreement</span>
          <span className="verdict-badge verdict-na">RESULT NOT AVAILABLE</span>
        </div>
        <p>
          No valence-arousal-dominance model or training data exists in this codebase.
          No human annotation process or inter-annotator agreement statistic exists for
          this project's own emotion taxonomy. Both would require new data-collection
          scope, not further evaluation of something already built — stated plainly
          rather than fabricated.
        </p>
      </div>

      <div className="card finding-card">
        <div className="finding-card-head">
          <span className="display-md">NewsAPI real pilot</span>
          <span className="verdict-badge verdict-mixed">Data-volume limitation, not a broken pipeline</span>
        </div>
        <p>
          A real combined YouTube+NewsAPI pilot produced 98 real processed articles. A
          root cause was found and fixed: articles published on non-trading days
          (weekends) were being silently dropped by the trading-calendar join. The fix —
          mapping forward-only (never backward) to the next real trading day — was
          verified on real data: 3 raw dates (2 of them weekend) correctly collapsed into
          2 trading-day-aligned dates.
        </p>
        <p>
          The 0-usable-rows outcome didn't change, for a different, now precisely
          isolated reason: the leakage-safe rolling z-score warm-up mathematically
          requires ≥3 real daily observations, and this run produced 2. NewsAPI's ~29-day
          free-tier lookback cap limits a single collection run to only 1-2 unique
          trading days.{" "}
          <strong>This is a genuine additional-data-required limitation — not evidence
          the pipeline is broken.</strong> It would resolve with real elapsed time across
          multiple collection runs, or a second, more historically flexible source.
        </p>
      </div>

      <div className="card finding-card">
        <div className="finding-card-head">
          <span className="display-md">Affective proxy correlation with market behavior</span>
          <span className="verdict-badge verdict-na">RESULT NOT AVAILABLE</span>
        </div>
        <p>
          Considered and explicitly rejected as scientifically indefensible: real text
          data spans at most 2 trading days per collection run, and a correlation
          computed from 2 points is mathematically degenerate (always exactly ±1 or
          undefined). Reporting such a number, even heavily caveated, would create a
          false impression of a computed result where none is statistically meaningful —
          so none is shown.
        </p>
      </div>

      <p className="source-link">Source: docs/NLP_MODEL_BENCHMARK.md, docs/TEXT_MODALITY_PILOT.md</p>
    </div>
  );
}
