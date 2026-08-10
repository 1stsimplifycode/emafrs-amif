// Renders RAMA's real per-row fusion weights as horizontal bars. No charting library --
// these are exact softmax outputs from the trained model (see
// src/emafrs/models/fusion.py), not illustrative/approximate values, so a plain
// proportional bar is an honest, sufficient representation.
interface Props {
  weights: Record<string, number>;
}

const MODALITY_COLORS: Record<string, string> = {
  financial: "#3b82f6",
  microstructure: "#f59e0b",
  text: "#10b981",
  emotion: "#a855f7",
};

export function ModalityWeights({ weights }: Props) {
  const entries = Object.entries(weights).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) {
    return <p className="muted">No modality weights available for this prediction.</p>;
  }
  return (
    <div className="modality-weights">
      {entries.map(([name, value]) => (
        <div key={name} className="modality-row">
          <span className="modality-label">{name}</span>
          <div className="modality-bar-track">
            <div
              className="modality-bar-fill"
              style={{
                width: `${Math.round(value * 100)}%`,
                backgroundColor: MODALITY_COLORS[name] || "#6b7280",
              }}
            />
          </div>
          <span className="modality-pct">{(value * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}
