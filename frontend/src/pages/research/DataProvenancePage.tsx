import { DataSourceHealthPanel } from "../../components/DataSourceHealthPanel";
import "../../styles/research.css";

export function DataProvenancePage() {
  return (
    <div>
      <div className="research-page-head">
        <span className="eyebrow">Data Provenance</span>
        <h1 className="display-xl">Where the evidence actually comes from</h1>
        <p className="text-muted" style={{ maxWidth: 620, marginTop: 8 }}>
          Live configuration status for every real data source EMAFRS can draw on. A
          source reported "not configured" is simply skipped — never mocked or silently
          faked.
        </p>
      </div>
      <div className="card" style={{ padding: "var(--space-5)" }}>
        <DataSourceHealthPanel />
      </div>
    </div>
  );
}
