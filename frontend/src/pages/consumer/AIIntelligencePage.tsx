import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, ApiError } from "../../api";
import type { PredictionResponse } from "../../api";
import { EvidenceConstellation } from "../../components/shared/EvidenceConstellation";
import { modalityMeta, ALL_KNOWN_MODALITIES } from "../../lib/modalities";
import { directionWord, pct } from "../../lib/format";
import { useWatchlist } from "../../lib/watchlist";
import "./AIIntelligencePage.css";

const OUTLOOK_COLOR: Record<string, string> = {
  bullish: "var(--bullish)",
  bearish: "var(--bearish)",
  neutral: "var(--neutral)",
};

// The signature page: a system-level view of how EMAFRS actually reasons, built
// entirely from one real, live-computed prediction (the user picks which tracked ticker
// to focus on) rather than a fabricated "market-wide" synthesis the backend has no way
// to compute today.
export function AIIntelligencePage() {
  const { tickers } = useWatchlist();
  const [focus, setFocus] = useState(tickers[0] ?? "SPY");
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "not-computed" | "error">("loading");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    api
      .latestPrediction(focus)
      .then((p) => {
        if (!cancelled) {
          setPrediction(p);
          setStatus("ready");
        }
      })
      .catch((e) => {
        if (cancelled) return;
        if (e instanceof ApiError && e.status === 404) setStatus("not-computed");
        else {
          setError(String(e));
          setStatus("error");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [focus]);

  return (
    <div className="ai-intel-page">
      <div className="panel-section">
        <span className="eyebrow">AI Market Intelligence</span>
        <h1 className="display-xl">How EMAFRS is reasoning right now</h1>
        <p className="text-muted" style={{ maxWidth: 620, marginTop: 8 }}>
          A live, real look at what evidence fed into the current outlook for one asset —
          not a marketing summary of the whole market, which this system does not
          currently compute.
        </p>
        <div className="focus-picker">
          <span className="text-faint">Focus asset:</span>
          {tickers.map((t) => (
            <button
              key={t}
              className={`focus-chip ${t === focus ? "focus-chip-active" : ""}`}
              onClick={() => setFocus(t)}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {status === "loading" && <p className="text-faint">Loading real prediction data…</p>}
      {status === "not-computed" && (
        <div className="card asset-empty-card">
          <p>No prediction computed yet for {focus}.</p>
          <Link className="btn btn-primary" to={`/asset/${focus}`}>
            Go compute one
          </Link>
        </div>
      )}
      {status === "error" && <p className="error-banner">{error}</p>}

      {status === "ready" && prediction && (
        <>
          <section className="panel-section">
            <div className="section-head">
              <h2 className="display-md">Current regime</h2>
            </div>
            <div className="card regime-hero-card">
              <span className="display-xl">{prediction.regime_name}</span>
              <span className="text-faint">
                as detected by the Market Regime Detection Engine, as of {prediction.as_of_date}
              </span>
            </div>
          </section>

          <section className="panel-section">
            <div className="section-head">
              <h2 className="display-md">Multimodal context</h2>
              <p className="text-faint">Which evidence sources actually shaped this outlook, and which didn't.</p>
            </div>
            <div className="mm-context-grid">
              <div className="card mm-constellation-card">
                <EvidenceConstellation
                  weights={prediction.modality_weights}
                  outlookLabel={directionWord(prediction.direction_label)}
                  outlookColorVar={OUTLOOK_COLOR[prediction.direction_label] ?? "var(--neutral)"}
                />
              </div>
              <div className="mm-modality-notes">
                {ALL_KNOWN_MODALITIES.map((m) => {
                  const meta = modalityMeta(m);
                  const present = m in prediction.modality_weights;
                  return (
                    <div key={m} className="card mm-note-card">
                      <div className="mm-note-head">
                        <span style={{ color: meta.color }} className="mm-note-dot" />
                        <span className="display-md">{meta.short}</span>
                        <span className={`pill ${present ? "pill-signal" : "pill-neutral"}`}>
                          {present ? `${pct(prediction.modality_weights[m])} weight` : "not available"}
                        </span>
                      </div>
                      <p className="text-muted mm-note-desc">{meta.description}</p>
                      {m === "text" && (
                        <p className="text-faint mm-note-caveat">
                          Real-time text/affective evidence is not currently included in
                          live predictions — accumulated real data volume has been
                          insufficient in every run to date. See{" "}
                          <Link to="/research/affective">Research → Affective Computing</Link>.
                        </p>
                      )}
                      {m === "microstructure" && !present && (
                        <p className="text-faint mm-note-caveat">
                          Intraday microstructure data is only available for a recent
                          trading window; this prediction fell outside that window.
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </section>

          <section className="panel-section">
            <div className="section-head">
              <h2 className="display-md">Signal consistency</h2>
            </div>
            <div className="card mm-consistency-card">
              <p>
                Not currently computed for live predictions. EMAFRS's cross-modal
                conflict-detection engine (comparing whether financial, market-behavior,
                and news evidence point the same direction) is implemented and unit-tested
                in the research codebase, but has never been evaluated against real
                simultaneous multimodal signals — see{" "}
                <Link to="/research/explainability">Research → Explainability</Link> for the
                exact status. Showing a consistency score here without that would be a
                number this system cannot yet back up.
              </p>
            </div>
          </section>

          <section className="panel-section">
            <div className="section-head">
              <h2 className="display-md">AI explanation</h2>
            </div>
            <div className="card mm-explanation-card">
              <p>{prediction.explanation_text}</p>
            </div>
          </section>

          <section className="panel-section">
            <div className="section-head">
              <h2 className="display-md">Limitations</h2>
              <p className="text-faint">What this specific outlook does not know.</p>
            </div>
            <ul className="mm-limitations-list">
              <li>
                Confidence ({pct(prediction.confidence)}) is the model's raw output — no
                tested calibration correction has been shown to unambiguously improve it
                (see <Link to="/research/calibration">Research → Calibration</Link>).
              </li>
              <li>
                The evidence attributions above describe what the model responded to, not
                a proven causal reason — faithfulness testing found them no more
                load-bearing than random feature removal (see{" "}
                <Link to="/research/explainability">Research → Explainability</Link>).
              </li>
              <li>
                Financial-only prediction accuracy, tested out-of-sample, is
                statistically indistinguishable from a class-frequency baseline — this is
                a genuinely hard prediction task (see{" "}
                <Link to="/research/performance">Research → Model Performance</Link>).
              </li>
            </ul>
          </section>
        </>
      )}
    </div>
  );
}
