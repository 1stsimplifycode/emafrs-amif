import "../../styles/research.css";

export function RegimeAnalysisPage() {
  return (
    <div>
      <div className="research-page-head">
        <span className="eyebrow">Regime Analysis</span>
        <h1 className="display-xl">Does the model actually adapt to market regime?</h1>
      </div>

      <div className="card finding-card">
        <div className="finding-card-head">
          <span className="display-md">Regime-probability leakage</span>
          <span className="verdict-badge verdict-fixed">Fixed &amp; verified</span>
        </div>
        <p>
          <strong>Original issue:</strong> regime probabilities were computed with
          hmmlearn's smoothed forward-backward posterior over the entire train+val+test
          sequence — a date's regime probability could, in principle, depend on
          observations that came after it, even though the HMM's fitted parameters were
          correctly train-only. This is a future-information pathway distinct from
          anything the project's existing leakage checks covered.
        </p>
        <p>
          <strong>Correction:</strong> replaced with a provably-exact forward-only causal
          filter (no backward pass). <strong>Verification:</strong> two independent
          proofs — the causal and smoothed posteriors must agree exactly at the last row
          of any sequence (where there's no future left to differ on), and the causal
          version is exactly truncation-invariant. Both hold. Measured practical impact
          on real data: negligible (&lt;1e-5) once the fitted model was held fixed, but the
          exact fix was adopted regardless since that property isn't guaranteed for every
          ticker.
        </p>
      </div>

      <div className="card finding-card">
        <div className="finding-card-head">
          <span className="display-md">RAMA regime-adaptivity</span>
          <span className="verdict-badge verdict-mixed">Statistically significant, practically modest</span>
        </div>
        <p>
          A permutation test (5,000 shuffles of regime labels, real test-set fusion
          weights) tests whether the observed regime-to-regime variation in the
          financial-modality fusion weight is distinguishable from chance.
        </p>
        <div className="metric-grid">
          <div className="metric-tile">
            <span className="metric-tile-label">Regime-to-regime range</span>
            <span className="metric-tile-value">≈ 1.6 pp</span>
          </div>
          <div className="metric-tile">
            <span className="metric-tile-label">Permutation shuffles</span>
            <span className="metric-tile-value">5,000</span>
          </div>
          <div className="metric-tile">
            <span className="metric-tile-label">p-value</span>
            <span className="metric-tile-value">&lt; 0.0002</span>
          </div>
          <div className="metric-tile">
            <span className="metric-tile-label">F-statistic</span>
            <span className="metric-tile-value">1644.6</span>
          </div>
        </div>
        <div className="research-table-wrap">
          <table className="research-table">
            <thead>
              <tr>
                <th>Regime</th>
                <th>N</th>
                <th>Mean financial weight</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>1</td>
                <td>27</td>
                <td>0.5019</td>
              </tr>
              <tr>
                <td>2</td>
                <td>71</td>
                <td>0.5117</td>
              </tr>
              <tr>
                <td>3</td>
                <td>45</td>
                <td>0.5086</td>
              </tr>
              <tr>
                <td>4</td>
                <td>17</td>
                <td>0.5179</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p>
          <strong>The correct characterization: statistically significant but modest
          regime-dependent modality weighting.</strong> Not "flat" (0 of 5,000 random
          shufflings produced a difference this extreme) and not "strongly adaptive"
          (the range is 1.6 percentage points). Two hypotheses for why the effect isn't
          larger were directly tested and rejected — excluding the gate parameter from
          weight decay had no measurable effect, and forcing 1,000 epochs of
          unregularized training moved the gate but regime-<em>invariantly</em>, while
          badly overfitting. The best-supported explanation is that this dataset's
          detected regimes genuinely don't carry large differential
          modality-reliability information — a property of the data, not a training bug,
          and not something more tuning would fix.
        </p>
      </div>

      <p className="source-link">Source: docs/EXPERIMENT_PROTOCOL.md, docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md row 19</p>
    </div>
  );
}
