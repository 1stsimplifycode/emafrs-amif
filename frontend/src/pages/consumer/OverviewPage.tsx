import { useMemo, useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { AssetPulseRow } from "../../components/consumer/AssetPulseRow";
import { useMultiTickerPredictions } from "../../lib/useMultiTickerPredictions";
import { useWatchlist } from "../../lib/watchlist";

export function OverviewPage() {
  const navigate = useNavigate();
  const { tickers, add, remove } = useWatchlist();
  const predictions = useMultiTickerPredictions(tickers);
  const [addValue, setAddValue] = useState("");
  const [heroQuery, setHeroQuery] = useState("");

  const ready = useMemo(
    () =>
      tickers
        .map((t) => ({ ticker: t, state: predictions[t] }))
        .filter((x) => x.state?.status === "ready") as {
        ticker: string;
        state: Extract<(typeof predictions)[string], { status: "ready" }>;
      }[],
    [tickers, predictions]
  );

  const bullishCount = ready.filter((r) => r.state.prediction.direction_label === "bullish").length;
  const bearishCount = ready.filter((r) => r.state.prediction.direction_label === "bearish").length;
  const regimeCounts = ready.reduce<Record<string, number>>((acc, r) => {
    acc[r.state.prediction.regime_name] = (acc[r.state.prediction.regime_name] ?? 0) + 1;
    return acc;
  }, {});
  const dominantRegime = Object.entries(regimeCounts).sort((a, b) => b[1] - a[1])[0]?.[0];

  const recentIntelligence = [...ready]
    .sort((a, b) => new Date(b.state.prediction.created_at).getTime() - new Date(a.state.prediction.created_at).getTime())
    .slice(0, 3);

  function submitHero(e: FormEvent) {
    e.preventDefault();
    const t = heroQuery.trim().toUpperCase();
    if (t) navigate(`/asset/${t}`);
  }

  function submitAdd(e: FormEvent) {
    e.preventDefault();
    add(addValue);
    setAddValue("");
  }

  return (
    <div className="overview-page">
      <section className="hero">
        <span className="eyebrow">EMAFRS — Explainable Multimodal Affective Financial Reasoning</span>
        <h1 className="display-xl">Understand the market beyond the numbers.</h1>
        <p className="text-muted hero-sub">
          EMAFRS doesn't just hand you a prediction. It shows you the evidence behind it,
          and exactly how trustworthy that evidence actually is.
        </p>
        <form className="hero-search" onSubmit={submitHero}>
          <input
            className="input hero-search-input"
            value={heroQuery}
            onChange={(e) => setHeroQuery(e.target.value)}
            placeholder="Explore an asset — e.g. SPY, NVDA, MSFT"
          />
          <button className="btn btn-primary" type="submit">
            Explore
          </button>
        </form>
      </section>

      <section className="panel-section">
        <div className="section-head">
          <h2 className="display-md">Market pulse</h2>
          <p className="text-faint">What's happening right now, across the assets you're tracking.</p>
        </div>
        {ready.length > 0 ? (
          <div className="pulse-summary-row">
            <div className="pulse-stat">
              <span className="pulse-stat-value numeric" style={{ color: "var(--bullish)" }}>
                {bullishCount}
              </span>
              <span className="text-faint">bullish outlooks</span>
            </div>
            <div className="pulse-stat">
              <span className="pulse-stat-value numeric" style={{ color: "var(--bearish)" }}>
                {bearishCount}
              </span>
              <span className="text-faint">bearish outlooks</span>
            </div>
            {dominantRegime && (
              <div className="pulse-stat">
                <span className="pulse-stat-value">{dominantRegime}</span>
                <span className="text-faint">most common regime</span>
              </div>
            )}
          </div>
        ) : (
          <p className="text-faint">
            No predictions computed yet for your tracked tickers — open one below to compute a real one.
          </p>
        )}
      </section>

      {recentIntelligence.length > 0 && (
        <section className="panel-section">
          <div className="section-head">
            <h2 className="display-md">AI intelligence</h2>
            <p className="text-faint">What EMAFRS is seeing right now — grounded in real model output, not templated copy.</p>
          </div>
          <div className="intelligence-cards">
            {recentIntelligence.map((r) => (
              <div key={r.ticker} className="card intelligence-card">
                <div className="intelligence-card-head">
                  <span className="eyebrow">{r.ticker}</span>
                  <span className="text-faint">{r.state.prediction.as_of_date}</span>
                </div>
                <p>{r.state.prediction.explanation_text}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="panel-section">
        <div className="section-head">
          <h2 className="display-md">Tracked assets</h2>
          <p className="text-faint">A local list of tickers you're following — add or remove any time.</p>
        </div>
        <div className="card watchlist-card">
          {tickers.map((t) => (
            <AssetPulseRow key={t} ticker={t} state={predictions[t] ?? { status: "loading" }} onRemove={() => remove(t)} />
          ))}
          <form className="watchlist-add" onSubmit={submitAdd}>
            <input
              className="input"
              value={addValue}
              onChange={(e) => setAddValue(e.target.value)}
              placeholder="Add a ticker to track"
            />
            <button className="btn" type="submit">
              Add
            </button>
          </form>
        </div>
      </section>
    </div>
  );
}
