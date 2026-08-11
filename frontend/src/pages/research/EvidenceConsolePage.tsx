import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../../api";
import type { PredictionResponse } from "../../api";
import { ModalityWeights } from "../../components/ModalityWeights";
import { signed } from "../../lib/format";
import "../../styles/research.css";

// The technical destination of the consumer product's "View evidence" link -- keyed by
// the real prediction ID from the real backend. There is no get-by-id endpoint, so this
// fetches the ticker's real history and finds the matching record client-side; nothing
// here is re-derived or fabricated, only filtered from an already-real response.
export function EvidenceConsolePage() {
  const { ticker = "", predictionId = "" } = useParams();
  const [record, setRecord] = useState<PredictionResponse | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "not-found" | "error">("loading");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    api
      .predictionHistory(ticker, 200)
      .then((history) => {
        if (cancelled) return;
        const match = history.find((r) => String(r.id) === predictionId);
        if (match) {
          setRecord(match);
          setStatus("ready");
        } else {
          setStatus("not-found");
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setError(String(e));
          setStatus("error");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [ticker, predictionId]);

  return (
    <div>
      <div className="research-page-head">
        <span className="eyebrow">Evidence Console</span>
        <h1 className="display-xl">
          {ticker} · prediction #{predictionId}
        </h1>
        <p className="text-faint">
          Full technical record behind the consumer product's outlook for this
          prediction — same real data, unabridged.
        </p>
      </div>

      {status === "loading" && <p className="text-faint">Loading…</p>}
      {status === "not-found" && (
        <p className="error-banner">
          No prediction #{predictionId} found in {ticker}'s persisted history (it may have
          been superseded — Render's free tier does not persist storage across restarts;
          see Data Provenance).
        </p>
      )}
      {status === "error" && <p className="error-banner">{error}</p>}

      {status === "ready" && record && (
        <>
          <div className="metric-grid">
            <div className="metric-tile">
              <span className="metric-tile-label">Direction</span>
              <span className="metric-tile-value">{record.direction_label}</span>
            </div>
            <div className="metric-tile">
              <span className="metric-tile-label">Confidence</span>
              <span className="metric-tile-value">{(record.confidence * 100).toFixed(2)}%</span>
            </div>
            <div className="metric-tile">
              <span className="metric-tile-label">Regime</span>
              <span className="metric-tile-value">{record.regime_name} ({record.regime})</span>
            </div>
            <div className="metric-tile">
              <span className="metric-tile-label">As of</span>
              <span className="metric-tile-value" style={{ fontSize: "1rem" }}>{record.as_of_date}</span>
            </div>
            <div className="metric-tile">
              <span className="metric-tile-label">Model version</span>
              <span className="metric-tile-value" style={{ fontSize: "1rem" }}>{record.model_version_id ?? "—"}</span>
            </div>
            <div className="metric-tile">
              <span className="metric-tile-label">Computed at</span>
              <span className="metric-tile-value" style={{ fontSize: "0.85rem" }}>
                {new Date(record.created_at).toLocaleString()}
              </span>
            </div>
          </div>

          <div className="card finding-card">
            <span className="display-md">Modality fusion weights (RAMA, exact)</span>
            <div style={{ marginTop: 12 }}>
              <ModalityWeights weights={record.modality_weights} />
            </div>
            <p className="text-faint" style={{ marginTop: 12 }}>
              Interpretation guidance: <Link to="/research/regime">Research → Regime Analysis</Link>
            </p>
          </div>

          <div className="card finding-card">
            <span className="display-md">Integrated Gradients attributions</span>
            <div className="research-table-wrap">
              <table className="research-table">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Attribution</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(record.top_attributions)
                    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                    .map(([name, val]) => (
                      <tr key={name}>
                        <td>{name}</td>
                        <td className={val >= 0 ? "attr-positive" : "attr-negative"}>{signed(val)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
            <p className="text-faint" style={{ marginTop: 12 }}>
              Faithfulness caveat: <Link to="/research/explainability">Research → Explainability</Link>
            </p>
          </div>

          {record.counterfactual && (
            <div className="card finding-card">
              <span className="display-md">Counterfactual</span>
              <p>
                {record.counterfactual.feature_name}: {record.counterfactual.original_value.toFixed(4)} →{" "}
                {record.counterfactual.counterfactual_value?.toFixed(4)} (Δ={" "}
                {signed(record.counterfactual.delta ?? 0, 4)}) flips the prediction.
              </p>
            </div>
          )}

          <div className="card finding-card">
            <span className="display-md">Raw explanation text (NLG)</span>
            <p>{record.explanation_text}</p>
          </div>

          <Link className="btn" to={`/asset/${ticker}`}>
            ← Back to consumer profile
          </Link>
        </>
      )}
    </div>
  );
}
