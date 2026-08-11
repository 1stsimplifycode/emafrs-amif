import { Link } from "react-router-dom";
import { ALL_KNOWN_MODALITIES, modalityMeta } from "../../lib/modalities";
import { pct } from "../../lib/format";
import "./EvidenceQualityPanel.css";

interface Props {
  confidence: number;
  modalityWeights: Record<string, number>;
  signalConsistency: number | null;
}

// Central EMAFRS product idea: prediction confidence is NOT the same thing as "how much
// should you trust this." This panel deliberately keeps four distinct questions visually
// separate rather than folding them into one score. Confidence and availability are
// always real per-prediction API values. Consistency is now ALSO a real, live-computed
// number (src/emafrs/risk/cross_modal_conflict.py, derived from this specific
// prediction's own Integrated Gradients attributions -- see training_pipeline.py) when
// two or more modalities actually contributed evidence; when only one did, it is
// mathematically undefined (nothing to compare), stated as such, never a fabricated
// score and never a blanket "not computed" claim that ignores what this prediction
// actually has.
export function EvidenceQualityPanel({ confidence, modalityWeights, signalConsistency }: Props) {
  const present = Object.keys(modalityWeights);
  const missing = ALL_KNOWN_MODALITIES.filter((m) => !present.includes(m));

  return (
    <div className="evidence-quality">
      <div className="eq-row">
        <div className="eq-label-group">
          <span className="eq-title">Prediction confidence</span>
          <span className="eq-help">The model's own probability for this outcome.</span>
        </div>
        <div className="eq-meter">
          <div className="eq-meter-track">
            <div className="eq-meter-fill" style={{ width: pct(confidence) }} />
          </div>
          <span className="eq-meter-value numeric">{pct(confidence)}</span>
        </div>
      </div>

      <div className="eq-row">
        <div className="eq-label-group">
          <span className="eq-title">Evidence availability</span>
          <span className="eq-help">
            {present.length} of {ALL_KNOWN_MODALITIES.length} known evidence sources contributed to this
            prediction.
          </span>
        </div>
        <div className="eq-modality-chips">
          {ALL_KNOWN_MODALITIES.map((m) => {
            const meta = modalityMeta(m);
            const has = present.includes(m);
            return (
              <span key={m} className={`eq-chip ${has ? "eq-chip-on" : "eq-chip-off"}`}>
                {has ? "✓" : "—"} {meta.short}
              </span>
            );
          })}
        </div>
      </div>

      <div className="eq-row">
        <div className="eq-label-group">
          <span className="eq-title">Evidence consistency</span>
          <span className="eq-help">Whether the available evidence sources agree or conflict, for THIS prediction.</span>
        </div>
        {signalConsistency !== null ? (
          <div className="eq-consistency">
            <span
              className={
                signalConsistency > 0
                  ? "pill pill-bullish"
                  : signalConsistency < 0
                    ? "pill pill-bearish"
                    : "pill pill-neutral"
              }
            >
              {signalConsistency > 0 ? "Agree" : signalConsistency < 0 ? "Conflict" : "Ambiguous"}
            </span>
            <span className="numeric eq-consistency-value">{signalConsistency.toFixed(2)}</span>
            <span className="text-faint eq-consistency-caption">
              Computed live from this prediction's own attributions across its{" "}
              {present.length} evidence sources (+1 = full agreement, −1 = full conflict). Not a
              causal claim about the market — see{" "}
              <Link to="/research/explainability">Research → Explainability</Link>.
            </span>
          </div>
        ) : (
          <div className="eq-unavailable">
            Not computable for this prediction — only {present.length} evidence source
            {present.length === 1 ? "" : "s"} contributed, and consistency requires at least two
            to compare against each other. See{" "}
            <Link to="/research/explainability">Research → Explainability</Link>.
          </div>
        )}
      </div>

      {missing.length > 0 && (
        <div className="eq-row">
          <div className="eq-label-group">
            <span className="eq-title">What's missing</span>
            <span className="eq-help">Known evidence sources not reflected in this prediction.</span>
          </div>
          <ul className="eq-missing-list">
            {missing.map((m) => (
              <li key={m}>
                <strong>{modalityMeta(m).short}:</strong> {modalityMeta(m).description}{" "}
                Not available for this prediction — see{" "}
                <Link to="/research/data">Research → Data Provenance</Link> for real, measured coverage.
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
