import type { PredictionResponse } from "../api";
import { ModalityWeights } from "./ModalityWeights";
import { TrustworthinessPanel } from "./TrustworthinessPanel";

interface Props {
  prediction: PredictionResponse;
}

const DIRECTION_COLORS: Record<string, string> = {
  bullish: "#16a34a",
  bearish: "#dc2626",
  neutral: "#6b7280",
};

export function PredictionCard({ prediction: p }: Props) {
  const attributionEntries = Object.entries(p.top_attributions).sort(
    (a, b) => Math.abs(b[1]) - Math.abs(a[1])
  );

  return (
    <div className="prediction-card">
      <div className="prediction-header">
        <div>
          <h2>{p.ticker}</h2>
          <span className="muted">as of {p.as_of_date}</span>
        </div>
        <div
          className="direction-badge"
          style={{ backgroundColor: DIRECTION_COLORS[p.direction_label] || "#6b7280" }}
        >
          {p.direction_label.toUpperCase()} — {(p.confidence * 100).toFixed(0)}% confidence
        </div>
      </div>

      <div className="prediction-grid">
        <section>
          <h3>Market regime</h3>
          <p className="regime-value">
            {p.regime_name} <span className="muted">(regime {p.regime})</span>
          </p>

          <h3>Modality contribution</h3>
          <ModalityWeights weights={p.modality_weights} />

          {p.manipulation_risk !== null && (
            <>
              <h3>Manipulation risk</h3>
              <p>{p.manipulation_risk.toFixed(3)}</p>
            </>
          )}
        </section>

        <section>
          <h3>Model-attributed evidence (Integrated Gradients)</h3>
          <ul className="attribution-list">
            {attributionEntries.map(([name, val]) => (
              <li key={name}>
                <span className="feature-name">{name}</span>
                <span className={val >= 0 ? "attr-positive" : "attr-negative"}>
                  {val >= 0 ? "+" : ""}
                  {val.toFixed(4)}
                </span>
              </li>
            ))}
          </ul>

          {p.counterfactual && (
            <>
              <h3>Counterfactual</h3>
              <p className="counterfactual-text">
                Had <strong>{p.counterfactual.feature_name}</strong> differed by{" "}
                <strong>
                  {p.counterfactual.delta !== null && p.counterfactual.delta >= 0 ? "+" : ""}
                  {p.counterfactual.delta?.toFixed(3)}
                </strong>{" "}
                (from {p.counterfactual.original_value.toFixed(3)} to{" "}
                {p.counterfactual.counterfactual_value?.toFixed(3)}), the model's prediction
                would have changed.
              </p>
            </>
          )}
        </section>
      </div>

      <section>
        <h3>Explanation</h3>
        <p className="explanation-text">{p.explanation_text}</p>
      </section>

      <TrustworthinessPanel />
    </div>
  );
}
