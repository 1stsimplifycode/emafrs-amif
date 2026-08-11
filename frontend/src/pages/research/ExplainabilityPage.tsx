import "../../styles/research.css";

export function ExplainabilityPage() {
  return (
    <div>
      <div className="research-page-head">
        <span className="eyebrow">Explainability</span>
        <h1 className="display-xl">Can the model's attributions actually be trusted?</h1>
      </div>

      <div className="card finding-card">
        <div className="finding-card-head">
          <span className="display-md">Completeness (Integrated Gradients' own theoretical guarantee)</span>
          <span className="verdict-badge verdict-fixed">Holds exactly</span>
        </div>
        <p>
          IG's completeness axiom (attributions must sum to f(x) − f(baseline)) is
          checked directly against real trained-model output — residual &lt; 1e-3 for
          both fusion architectures. This is a property of the attribution mechanism
          working correctly, not a claim about faithfulness (below).
        </p>
      </div>

      <div className="card finding-card">
        <div className="finding-card-head">
          <span className="display-md">Faithfulness — four independent methods</span>
          <span className="verdict-badge verdict-weak">Weak, confirmed</span>
        </div>
        <p>
          Faithfulness measures whether the top-attributed features are actually more
          load-bearing to the prediction than random ones. Tested across two IG
          baselines, a pure local-gradient method (Gradient×Input), and a gradient-free
          removal method (Occlusion) — deliberately, to see whether a weak result was an
          artifact of IG specifically.
        </p>
        <div className="research-table-wrap">
          <table className="research-table">
            <thead>
              <tr>
                <th>Method</th>
                <th>Faithfulness</th>
                <th>Deletion AUC</th>
                <th>Insertion AUC</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>IG (zero baseline)</td>
                <td>−0.0004</td>
                <td>0.4143</td>
                <td>0.4062</td>
              </tr>
              <tr>
                <td>IG (random-sample baseline)</td>
                <td>−0.0008</td>
                <td>0.4140</td>
                <td>0.4039</td>
              </tr>
              <tr>
                <td>Gradient × Input</td>
                <td>+0.0020</td>
                <td>0.4137</td>
                <td>0.4048</td>
              </tr>
              <tr>
                <td>Occlusion (gradient-free)</td>
                <td>+0.0106</td>
                <td>0.4096</td>
                <td>0.4100</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p>
          <strong>
            Attributions can be generated, but current evidence indicates weak
            faithfulness.
          </strong>{" "}
          All four methods agree: faithfulness is statistically indistinguishable from
          zero, and deletion AUC ≈ insertion AUC for every method (a faithful ranking
          should show insertion clearly exceed deletion). Baseline choice was ruled out
          (two baselines, nearly identical results). A saturation check found gradient
          norms roughly halve moving from the baseline toward real inputs — real, but
          not the sole explanation, since a gradient-free method shows the same pattern.
        </p>
        <p>
          The best-supported explanation: this model's predicted probabilities don't vary
          sharply with any small feature subset — consistent with the near-chance
          predictive performance documented on Model Performance. This is treated as a
          property of the model/task, not a fixable explainability-tooling problem;
          further explainer-shopping (e.g. SHAP) would be expected to reproduce the same
          result.
        </p>
        <p className="text-faint">
          <strong>Precision correction, not a separate finding:</strong> the raw
          diagnostic run showed 100% of IG attribution mass on the financial modality,
          0% on microstructure. Verified before drawing any conclusion from it: only 1 of
          160 real test-set rows had real microstructure data available, and none of the
          60 randomly-sampled diagnostic rows happened to include that one row — so
          near-zero microstructure attribution is a direct, correct consequence of
          microstructure being absent from the sample, not evidence the model ignores it
          when genuinely present. The faithfulness result above is therefore a statement
          about the financial-only decision surface, not a multimodal fusion problem.
        </p>
      </div>

      <div className="card finding-card">
        <div className="finding-card-head">
          <span className="display-md">Stability, sparsity, counterfactual validity</span>
          <span className="verdict-badge verdict-fixed">Consistently strong</span>
        </div>
        <div className="metric-grid">
          <div className="metric-tile">
            <span className="metric-tile-label">Stability</span>
            <span className="metric-tile-value">0.95–0.97</span>
          </div>
          <div className="metric-tile">
            <span className="metric-tile-label">Counterfactual validity</span>
            <span className="metric-tile-value">73–80%</span>
          </div>
        </div>
        <p>
          Attributions don't flip wildly under small input noise, and a
          prediction-flipping counterfactual genuinely exists for most sampled rows —
          both positive, real findings reported alongside the faithfulness weakness
          rather than instead of it.
        </p>
      </div>

      <p className="source-link">Source: docs/XAI_EVALUATION.md, docs/_xai_diagnostic_results.md</p>
    </div>
  );
}
