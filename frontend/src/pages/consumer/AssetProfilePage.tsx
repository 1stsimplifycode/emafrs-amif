import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../../api";
import type { PredictionResponse } from "../../api";
import { EvidenceConstellation } from "../../components/shared/EvidenceConstellation";
import { EvidenceQualityPanel } from "../../components/shared/EvidenceQualityPanel";
import { RegimeTimeline } from "../../components/shared/RegimeTimeline";
import { StagedProgressIndicator } from "../../components/shared/StagedProgressIndicator";
import { TrustworthinessPanel } from "../../components/TrustworthinessPanel";
import { friendlyErrorMessage } from "../../lib/errors";
import { featureLabel } from "../../lib/featureLabels";
import { directionPillClass, directionWord, pct, signed } from "../../lib/format";
import { usePrediction } from "../../lib/usePrediction";
import "./AssetProfilePage.css";

const OUTLOOK_COLOR: Record<string, string> = {
  bullish: "var(--bullish)",
  bearish: "var(--bearish)",
  neutral: "var(--neutral)",
};

export function AssetProfilePage() {
  const { ticker = "" } = useParams();
  const { status, prediction, isRevalidating, refresh } = usePrediction(ticker);
  const [history, setHistory] = useState<PredictionResponse[]>([]);
  const [computing, setComputing] = useState(false);
  const [computeError, setComputeError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.predictionHistory(ticker).then((h) => !cancelled && setHistory(h)).catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [ticker, prediction?.id]);

  async function compute() {
    setComputing(true);
    setComputeError(null);
    try {
      await refresh();
      const h = await api.predictionHistory(ticker).catch(() => []);
      setHistory(h);
    } catch {
      setComputeError(friendlyErrorMessage(`${ticker}'s outlook`));
    } finally {
      setComputing(false);
    }
  }

  return (
    <div className="asset-page">
      <div className="asset-page-head">
        <span className="eyebrow">AI Financial Intelligence Profile</span>
        <h1 className="display-xl">{ticker}</h1>
      </div>

      {status === "loading" && <p className="text-faint">Loading real prediction data…</p>}

      {status === "not-computed" && (
        <div className="card asset-empty-card">
          <p>No prediction has been computed yet for {ticker}.</p>
          <button className="btn btn-primary" onClick={compute} disabled={computing}>
            {computing ? "Training a real model…" : "Compute AI outlook"}
          </button>
          <StagedProgressIndicator active={computing} />
          {computeError && <p className="error-banner" style={{ marginTop: 12 }}>{computeError}</p>}
        </div>
      )}

      {status === "error" && <p className="error-banner">{friendlyErrorMessage(`${ticker}'s prediction`)}</p>}

      {status === "ready" && prediction && (
        <div key={prediction.id} className="fade-in-update">
          <section className="asset-outlook-row">
            <div className="card asset-outlook-card">
              <span className="eyebrow">
                AI outlook
                {isRevalidating && <span className="revalidating-dot" title="Confirming with a live fetch…" />}
              </span>
              <div className="asset-outlook-value">
                <span className={directionPillClass(prediction.direction_label)} style={{ fontSize: "0.95rem", padding: "6px 16px" }}>
                  {directionWord(prediction.direction_label)}
                </span>
              </div>
              <div className="asset-confidence-row">
                <span className="text-faint">Confidence</span>
                <span className="numeric asset-confidence-value">{pct(prediction.confidence)}</span>
              </div>
              <div className="asset-meta-row text-faint">
                As of {prediction.as_of_date} · {prediction.regime_name} regime
              </div>
              <button className="btn btn-ghost" style={{ marginTop: 12 }} onClick={compute} disabled={computing}>
                {computing ? "Recomputing…" : "Recompute with latest data"}
              </button>
              <StagedProgressIndicator active={computing} />
              {computeError && <p className="error-banner" style={{ marginTop: 12 }}>{computeError}</p>}
            </div>

            <div className="card asset-constellation-card">
              <span className="eyebrow">What is driving the outlook?</span>
              <EvidenceConstellation
                weights={prediction.modality_weights}
                outlookLabel={directionWord(prediction.direction_label)}
                outlookColorVar={OUTLOOK_COLOR[prediction.direction_label] ?? "var(--neutral)"}
              />
            </div>
          </section>

          <section className="panel-section-tight">
            <h2 className="display-md">Why?</h2>
            <div className="card asset-why-card">
              <p className="asset-explanation-text">{prediction.explanation_text}</p>
              {Object.keys(prediction.top_attributions).length > 0 && (
                <div className="asset-attribution-grid">
                  {Object.entries(prediction.top_attributions)
                    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                    .map(([name, val]) => (
                      <div key={name} className="asset-attribution-item">
                        <span className="text-muted">{featureLabel(name)}</span>
                        <span className={val >= 0 ? "attr-positive numeric" : "attr-negative numeric"}>{signed(val)}</span>
                      </div>
                    ))}
                </div>
              )}
            </div>
          </section>

          <section className="panel-section-tight">
            <h2 className="display-md">Evidence quality</h2>
            <div className="card">
              <EvidenceQualityPanel
                confidence={prediction.confidence}
                modalityWeights={prediction.modality_weights}
                signalConsistency={prediction.signal_consistency}
              />
            </div>
          </section>

          {prediction.counterfactual && (
            <section className="panel-section-tight">
              <h2 className="display-md">What could change this outlook?</h2>
              <div className="card asset-counterfactual-card">
                <p>
                  Had <strong>{featureLabel(prediction.counterfactual.feature_name)}</strong> differed by{" "}
                  <strong className="numeric">{signed(prediction.counterfactual.delta ?? 0, 3)}</strong> (from{" "}
                  <span className="numeric">{prediction.counterfactual.original_value.toFixed(3)}</span> to{" "}
                  <span className="numeric">{prediction.counterfactual.counterfactual_value?.toFixed(3)}</span>), the
                  model's prediction would have flipped. This describes model sensitivity to this one input, not a
                  causal claim about what will happen in the market.
                </p>
              </div>
            </section>
          )}

          {history.length > 0 && (
            <section className="panel-section-tight">
              <h2 className="display-md">Regime history</h2>
              <div className="card">
                <RegimeTimeline history={history} />
              </div>
            </section>
          )}

          <TrustworthinessPanel />

          <div className="asset-evidence-link">
            <Link className="btn" to={`/research/evidence/${ticker}/${prediction.id}`}>
              View full evidence in Research Console →
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
