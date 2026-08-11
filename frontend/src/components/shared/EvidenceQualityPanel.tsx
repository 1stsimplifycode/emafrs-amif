import { Link } from "react-router-dom";
import { ALL_KNOWN_MODALITIES, modalityMeta } from "../../lib/modalities";
import { pct } from "../../lib/format";
import "./EvidenceQualityPanel.css";

interface Props {
  confidence: number;
  modalityWeights: Record<string, number>;
}

// Central EMAFRS product idea: prediction confidence is NOT the same thing as "how much
// should you trust this." This panel deliberately keeps four distinct questions visually
// separate rather than folding them into one score. Each answer is either a real number
// derived from the actual API response, or an explicit "not currently available" -- never
// a fabricated consistency/trust score.
export function EvidenceQualityPanel({ confidence, modalityWeights }: Props) {
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
          <span className="eq-help">Whether the available evidence sources agree or conflict.</span>
        </div>
        <div className="eq-unavailable">
          Not currently computed for live predictions — cross-modal consistency checking
          is implemented and tested in the research codebase but has not yet been wired
          into this endpoint. See <Link to="/research/explainability">Research → Explainability</Link>.
        </div>
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
