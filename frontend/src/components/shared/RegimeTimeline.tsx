import type { PredictionResponse } from "../../api";
import "./RegimeTimeline.css";

interface Props {
  history: PredictionResponse[];
}

const REGIME_COLOR: Record<string, string> = {
  Low: "var(--bullish)",
  Medium: "var(--caution)",
  High: "var(--structural)",
  Extreme: "var(--bearish)",
};

// A real, if simple, timeline of regime + direction across every PERSISTED prediction
// for this ticker (oldest to newest, left to right) -- not a price chart (this dashboard
// has no live price feed), and not fabricated: it stops exactly where real history stops.
export function RegimeTimeline({ history }: Props) {
  if (history.length === 0) {
    return <p className="text-faint">No prediction history yet for this ticker.</p>;
  }
  const chronological = [...history].reverse();

  return (
    <div className="regime-timeline">
      <div className="regime-timeline-track">
        {chronological.map((p) => (
          <div key={p.id} className="regime-timeline-cell" title={`${p.as_of_date}: ${p.regime_name} regime, ${p.direction_label}`}>
            <div
              className="regime-timeline-bar"
              style={{ background: REGIME_COLOR[p.regime_name] ?? "var(--neutral)" }}
            />
          </div>
        ))}
      </div>
      <div className="regime-timeline-labels">
        <span className="text-faint">{chronological[0].as_of_date}</span>
        <span className="text-faint">{chronological[chronological.length - 1].as_of_date}</span>
      </div>
      <div className="regime-timeline-legend">
        {Object.entries(REGIME_COLOR).map(([name, color]) => (
          <span key={name} className="regime-legend-item">
            <span className="regime-legend-swatch" style={{ background: color }} />
            {name}
          </span>
        ))}
      </div>
    </div>
  );
}
