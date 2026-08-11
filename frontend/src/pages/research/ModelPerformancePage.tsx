import "../../styles/research.css";

export function ModelPerformancePage() {
  return (
    <div>
      <div className="research-page-head">
        <span className="eyebrow">Model Performance</span>
        <h1 className="display-xl">Predictive competence, tested honestly</h1>
      </div>

      <div className="card finding-card">
        <div className="finding-card-head">
          <span className="display-md">Financial-only walk-forward (5 real out-of-sample folds, SPY)</span>
          <span className="verdict-badge verdict-weak">Weak — null result</span>
        </div>
        <p>
          Three models (opaque XGBoost, an explainable logistic regression, and a
          single-modality neural fusion network) were tested with true walk-forward
          validation — expanding training windows, non-overlapping test folds, never
          shuffled.
        </p>
        <div className="research-table-wrap">
          <table className="research-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Mean accuracy</th>
                <th>95% CI</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>A — opaque XGBoost</td>
                <td>0.384</td>
                <td>[0.349, 0.419]</td>
              </tr>
              <tr>
                <td>B — explainable logistic regression</td>
                <td>0.390</td>
                <td>[0.374, 0.406]</td>
              </tr>
              <tr>
                <td>NN — single-modality fusion</td>
                <td>0.402</td>
                <td>[0.379, 0.426]</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p>
          None of the three is statistically distinguishable from the others (paired
          t-test, all p &gt; 0.16). On a 3-class problem, a class-frequency baseline
          lands around 33-38%. This is a genuine, unadjusted negative result, consistent
          with the market being efficient enough that 1-day-ahead direction is not
          reliably predictable from technical features alone.
        </p>
      </div>

      <div className="card finding-card">
        <div className="finding-card-head">
          <span className="display-md">Multimodal single-split (financial + microstructure, most recent real run)</span>
          <span className="verdict-badge verdict-mixed">Insufficient evidence for a stronger claim</span>
        </div>
        <div className="research-table-wrap">
          <table className="research-table">
            <thead>
              <tr>
                <th>Regime</th>
                <th>N</th>
                <th>Accuracy</th>
                <th>Macro F1</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>1 — Low</td>
                <td>27</td>
                <td>0.407</td>
                <td>0.193</td>
              </tr>
              <tr>
                <td>2 — Medium</td>
                <td>71</td>
                <td>0.408</td>
                <td>0.233</td>
              </tr>
              <tr>
                <td>3 — High</td>
                <td>45</td>
                <td>0.422</td>
                <td>0.273</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p>
          This reaches 41-43% on a single split, but is NOT walk-forward-able: real
          intraday microstructure data only covers a recent ~5-7 trading-day window (a
          real data-provider limitation, not a design choice), so it cannot be
          repeated across multiple historical folds without fabricating "as of"
          availability for dates that never had real intraday data.
        </p>
      </div>

      <p className="source-link">Source: docs/EXPERIMENT_PROTOCOL.md</p>
    </div>
  );
}
