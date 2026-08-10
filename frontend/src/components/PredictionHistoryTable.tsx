import type { PredictionResponse } from "../api";

interface Props {
  history: PredictionResponse[];
}

export function PredictionHistoryTable({ history }: Props) {
  if (history.length === 0) {
    return <p className="muted">No prediction history yet for this ticker.</p>;
  }
  return (
    <table className="history-table">
      <thead>
        <tr>
          <th>As of</th>
          <th>Direction</th>
          <th>Confidence</th>
          <th>Regime</th>
          <th>Computed at</th>
        </tr>
      </thead>
      <tbody>
        {history.map((p) => (
          <tr key={p.id}>
            <td>{p.as_of_date}</td>
            <td>{p.direction_label}</td>
            <td>{(p.confidence * 100).toFixed(0)}%</td>
            <td>{p.regime_name}</td>
            <td>{new Date(p.created_at).toLocaleString()}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
