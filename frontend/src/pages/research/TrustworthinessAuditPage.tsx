import "../../styles/research.css";

interface Row {
  dimension: string;
  classification: string;
  verdict: "fixed" | "weak" | "mixed" | "na";
}

const ROWS: Row[] = [
  { dimension: "Predictive performance (financial-only)", classification: "Weak — near market-efficient at this horizon", verdict: "weak" },
  { dimension: "Predictive performance (multimodal)", classification: "Insufficient evidence — single split only, no walk-forward", verdict: "mixed" },
  { dimension: "Calibration", classification: "Weak — no variant unambiguously beats uncalibrated across all metrics", verdict: "weak" },
  { dimension: "Data integrity / leakage", classification: "Moderate-to-strong — one real gap found and closed, everything else re-verified", verdict: "fixed" },
  { dimension: "Explainability — mechanism correctness", classification: "Strong — completeness axiom holds exactly", verdict: "fixed" },
  { dimension: "Explainability — faithfulness", classification: "Weak — confirmed via 4 independent attribution methods", verdict: "weak" },
  { dimension: "Explainability — stability / sparsity / counterfactual", classification: "Moderate-to-strong — consistent positive results", verdict: "fixed" },
  { dimension: "RAMA regime-adaptivity", classification: "Statistically real, practically modest (p<0.0002, ~1.6pp)", verdict: "mixed" },
  { dimension: "Affective computing — sentiment", classification: "Moderate — real ground truth, disclosed in-sample caveat", verdict: "mixed" },
  { dimension: "Affective computing — emotion / V-A-D / agreement", classification: "RESULT NOT AVAILABLE — no ground truth or capability exists", verdict: "na" },
  { dimension: "Cross-modal consistency", classification: "Insufficient evidence — implemented, never run on real simultaneous data", verdict: "mixed" },
  { dimension: "Missing-modality robustness", classification: "Moderate — graceful degradation observed (unplanned natural experiment)", verdict: "mixed" },
  { dimension: "Uncertainty (epistemic) / OOD detection", classification: "NOT EVALUATED — capability not implemented, not attempted this pass", verdict: "na" },
  { dimension: "Human trust", classification: "NOT EVALUATED — no human-subjects study exists", verdict: "na" },
];

const VERDICT_CLASS: Record<Row["verdict"], string> = {
  fixed: "verdict-fixed",
  weak: "verdict-weak",
  mixed: "verdict-mixed",
  na: "verdict-na",
};

const VERDICT_LABEL: Record<Row["verdict"], string> = {
  fixed: "Strong",
  weak: "Weak",
  mixed: "Mixed",
  na: "N/A",
};

export function TrustworthinessAuditPage() {
  return (
    <div>
      <div className="research-page-head">
        <span className="eyebrow">Trustworthiness Audit</span>
        <h1 className="display-xl">The full structured assessment</h1>
        <p className="text-muted" style={{ maxWidth: 640, marginTop: 8 }}>
          Every dimension below is either backed by a real, cited run; marked as a
          root-caused, corrected, and re-evaluated finding; or marked{" "}
          <strong>RESULT NOT AVAILABLE</strong> / <strong>NOT EVALUATED</strong> with the
          specific missing capability named. "Not evaluated" is never rounded up to
          "performed poorly," and "performed poorly" is never rounded down to "not
          available" — the distinctions are kept exact.
        </p>
      </div>

      <div className="research-table-wrap">
        <table className="research-table">
          <thead>
            <tr>
              <th>Dimension</th>
              <th>Classification</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((r) => (
              <tr key={r.dimension}>
                <td style={{ fontWeight: 600 }}>{r.dimension}</td>
                <td className="text-muted">{r.classification}</td>
                <td>
                  <span className={`verdict-badge ${VERDICT_CLASS[r.verdict]}`}>{VERDICT_LABEL[r.verdict]}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-faint" style={{ marginTop: "var(--space-4)" }}>
        No dimension is classified favorably to compensate for another being weak. Where
        evidence conflicted with an earlier published hypothesis — most notably RAMA's
        regime-adaptivity and calibration's metric-level disagreement — the hypothesis
        was revised in place rather than the new evidence being suppressed.
      </p>

      <p className="source-link">Source: docs/TRUSTWORTHINESS_AUDIT.md §M.6</p>
    </div>
  );
}
