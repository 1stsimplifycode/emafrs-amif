import { useState } from "react";
import "./App.css";
import { api, ApiError } from "./api";
import type { PredictionResponse } from "./api";
import { DataSourceHealthPanel } from "./components/DataSourceHealthPanel";
import { PredictionCard } from "./components/PredictionCard";
import { PredictionHistoryTable } from "./components/PredictionHistoryTable";

function App() {
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
    <div className="app">
      <header className="app-header">
        <h1>EMAFRS / AMIF Dashboard</h1>
        <p className="muted">
          Explainable Multimodal Affective Financial Reasoning System — research
          reconstruction. Every number below comes from a real, live-trained model or a
          real API call; nothing is a placeholder.
        </p>
      </header>

      <div className="controls">
        <input
          value={ticker}
          onChange={(e) => setTicker(e.target.value.toUpperCase())}
          placeholder="Ticker (e.g. SPY)"
        />
        <button onClick={() => loadLatest(ticker)} disabled={loading !== null}>
          {loading === "latest" ? "Loading..." : "Load latest"}
        </button>
        <button onClick={() => refresh(ticker)} disabled={loading !== null}>
          {loading === "refresh" ? "Training (~10-30s)..." : "Train & Predict"}
        </button>
      </div>

      {error && <p className="error">{error}</p>}

      {prediction && (
        <>
          <PredictionCard prediction={prediction} />
          <section className="panel">
            <h2>Prediction history</h2>
            <PredictionHistoryTable history={history} />
          </section>
        </>
      )}

      <section className="panel">
        <h2>Data source health</h2>
        <DataSourceHealthPanel />
      </section>

      <footer className="app-footer">
        <p className="muted">
          See the repository's <code>docs/</code> for full experimental results, ablation
          study, limitations, and reproducibility notes. This dashboard surfaces live model
          output; it does not itself constitute a research claim beyond what those docs
          already document.
        </p>
      </footer>
    </div>
  );
}

export default App;
