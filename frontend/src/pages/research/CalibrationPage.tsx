import "../../styles/research.css";

export function CalibrationPage() {
  return (
    <div>
      <div className="research-page-head">
        <span className="eyebrow">Calibration</span>
        <h1 className="display-xl">Is confidence trustworthy?</h1>
      </div>

      <div className="card finding-card">
        <div className="finding-card-head">
          <span className="display-md">Regime-dependent temperature scaling, real held-out test set</span>
          <span className="verdict-badge verdict-mixed">Metrics disagree — no unambiguous winner</span>
        </div>
        <p>
          Three calibration variants were tested against the uncalibrated model on the
          same real test split: a naive per-regime temperature, a single global
          temperature, and a shrinkage-regularized per-regime fit. All three metrics
          — ECE, Brier, and NLL — are reported, not just the one that looks best.
        </p>
        <div className="research-table-wrap">
          <table className="research-table">
            <thead>
              <tr>
                <th>Variant</th>
                <th>Brier</th>
                <th>ECE</th>
                <th>NLL</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Uncalibrated</td>
                <td>0.660</td>
                <td>0.031</td>
                <td>1.091</td>
              </tr>
              <tr>
                <td>Naive per-regime</td>
                <td>0.754</td>
                <td>0.123</td>
                <td>2.384</td>
              </tr>
              <tr>
                <td>Global-only (1 free parameter)</td>
                <td>0.666</td>
                <td>0.027</td>
                <td>1.106</td>
              </tr>
              <tr>
                <td>Shrinkage per-regime</td>
                <td>0.667</td>
                <td>0.032</td>
                <td>1.108</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p>
          Global-only calibration improves ECE (0.027 vs. 0.031) — but is actually
          slightly worse than doing nothing on both Brier (0.666 vs. 0.660) and NLL
          (1.106 vs. 1.091). <strong>No tested calibration variant is unambiguously
          better than the uncalibrated model across every metric.</strong> Reporting only
          the ECE improvement would have overstated the case for calibrating at all — so
          all three metrics are shown together, deliberately.
        </p>
        <p className="text-faint">
          Naive per-regime became dramatically worse on every metric — traced to a
          degenerate near-zero temperature (0.018) fit for the smallest regime group
          (n=27), consistent with small-sample overfitting.
        </p>
      </div>

      <p className="source-link">Source: docs/EXPERIMENT_PROTOCOL.md §Calibration</p>
    </div>
  );
}
