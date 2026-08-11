import { useState } from "react";
import { Link } from "react-router-dom";
import { api, ApiError } from "../../api";
import type { PredictionResponse } from "../../api";
import { PredictionCard } from "../../components/PredictionCard";
import { PredictionHistoryTable } from "../../components/PredictionHistoryTable";
import { StagedProgressIndicator } from "../../components/shared/StagedProgressIndicator";
import "../../styles/research.css";

const SUBPAGES = [
  { to: "/research/performance", label: "Model Performance", desc: "Walk-forward accuracy, significance tests, baselines." },
  { to: "/research/calibration", label: "Calibration", desc: "ECE, Brier, NLL — and where they disagree." },
  { to: "/research/regime", label: "Regime Analysis", desc: "RAMA's regime-adaptivity, root-caused and significance-tested." },
  { to: "/research/explainability", label: "Explainability", desc: "IG faithfulness, alternative attribution methods, saturation." },
  { to: "/research/affective", label: "Affective Computing", desc: "Sentiment validation, emotion taxonomy, real pilot data." },
  { to: "/research/data", label: "Data Provenance", desc: "Live data-source configuration and reachability." },
  { to: "/research/audit", label: "Trustworthiness Audit", desc: "The full structured assessment, dimension by dimension." },
  { to: "/research/methodology", label: "Methodology", desc: "Leakage audit, no-fabrication rule, how findings were verified." },
];

export function ResearchHomePage() {
  const [ticker, setTicker] = useState("SPY");
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [history, setHistory] = useState<PredictionResponse[]>([]);
  const [loading, setLoading] = useState<"latest" | "refresh" | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function loadLatest(t: string) {
    setLoading("latest");
    setError(null);
    try {
      const [pred, hist] = await Promise.all([
        api.latestPrediction(t).catch((e: ApiError) => {
          if (e.status === 404) return null;
          throw e;
        }),
        api.predictionHistory(t).catch(() => []),
      ]);
      setPrediction(pred);
      setHistory(hist);
      if (pred === null) {
        setError(`No prediction computed yet for ${t.toUpperCase()}. Click "Train & Predict" to compute one (takes ~10-30s).`);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(null);
    }
  }

  async function refresh(t: string) {
    setLoading("refresh");
    setError(null);
    try {
      const pred = await api.refreshPrediction(t);
      setPrediction(pred);
      const hist = await api.predictionHistory(t).catch(() => []);
      setHistory(hist);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(null);
    }
  }

  return (
    <div className="research-home">
      <div className="research-page-head">
        <span className="eyebrow">Research &amp; Trust workspace</span>
        <h1 className="display-xl">Can this system actually be trusted?</h1>
        <p className="text-muted" style={{ maxWidth: 640, marginTop: 8 }}>
          The technical evidence behind everything the product surface claims — real
          experiments, real numbers, honestly reported when they're weak. Same backend,
          same models, same data as the consumer product.
        </p>
      </div>

      <section className="panel-section">
        <div className="section-head">
          <h2 className="display-md">Live model console</h2>
          <p className="text-faint">Run the real training/inference pipeline directly and inspect full technical output.</p>
        </div>
        <div className="controls">
          <input value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} placeholder="Ticker (e.g. SPY)" className="input" />
          <button className="btn" onClick={() => loadLatest(ticker)} disabled={loading !== null}>
            {loading === "latest" ? "Loading..." : "Load latest"}
          </button>
          <button className="btn btn-primary" onClick={() => refresh(ticker)} disabled={loading !== null}>
            {loading === "refresh" ? "Training..." : "Train & Predict"}
          </button>
        </div>

        <StagedProgressIndicator active={loading === "refresh"} />

        {error && <p className="error-banner" style={{ marginTop: 12 }}>{error}</p>}

        {prediction && (
          <div key={prediction.id} className="fade-in-update">
            <PredictionCard prediction={prediction} />
            <div className="panel" style={{ marginTop: 20 }}>
              <h2 style={{ marginTop: 0 }}>Prediction history</h2>
              <PredictionHistoryTable history={history} />
            </div>
          </div>
        )}
      </section>

      <section className="panel-section">
        <div className="section-head">
          <h2 className="display-md">Explore the findings</h2>
        </div>
        <div className="research-nav-grid">
          {SUBPAGES.map((p) => (
            <Link key={p.to} to={p.to} className="card research-nav-card">
              <span className="display-md">{p.label}</span>
              <span className="text-faint">{p.desc}</span>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
