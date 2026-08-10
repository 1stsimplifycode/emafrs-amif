import { useEffect, useState } from "react";
import { api } from "../api";
import type { DataSourcesHealthResponse } from "../api";

// PDF/build brief requirement: dashboard must show API health and data freshness, and
// must never render stale data as if it were live. `checked_live` is surfaced directly
// so the user knows whether this is a config-only snapshot or an actual reachability probe.
export function DataSourceHealthPanel() {
  const [data, setData] = useState<DataSourcesHealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .dataSourcesHealth(false)
      .then(setData)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p className="muted">Checking data source configuration...</p>;
  if (error) return <p className="error">Failed to load data source health: {error}</p>;
  if (!data) return null;

  return (
    <div className="health-panel">
      <p className="muted">
        {data.checked_live
          ? "Live reachability probe (real network calls made)."
          : "Configuration status only -- no live network probe was made (pass live=true for that, sparingly: some providers charge real quota per probe)."}
      </p>
      <table className="health-table">
        <thead>
          <tr>
            <th>Provider</th>
            <th>Configured</th>
            <th>Reachable</th>
            <th>Last success</th>
          </tr>
        </thead>
        <tbody>
          {data.sources.map((s) => (
            <tr key={s.provider}>
              <td>{s.provider}</td>
              <td>
                <span className={s.configured ? "badge badge-ok" : "badge badge-off"}>
                  {s.configured ? "configured" : "not configured"}
                </span>
              </td>
              <td>
                {s.reachable === null ? (
                  <span className="muted">not probed</span>
                ) : s.reachable ? (
                  <span className="badge badge-ok">reachable</span>
                ) : (
                  <span className="badge badge-error">unreachable</span>
                )}
              </td>
              <td>{s.last_success_at ? new Date(s.last_success_at).toLocaleString() : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
