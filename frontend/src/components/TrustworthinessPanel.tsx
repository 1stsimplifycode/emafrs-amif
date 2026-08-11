// Surfaces this project's own real, measured trustworthiness findings (from a full
// diagnostic audit -- see docs/TRUSTWORTHINESS_AUDIT.md) directly alongside the numbers
// they qualify, rather than only showing attributions/confidence/fusion-weights as if
// they were self-evidently trustworthy. These figures are NOT computed live per-request
// (they are properties of the trained model/methodology in general, established via
// dedicated real evaluation scripts) -- each one is cited so it can be checked against
// the repo, not just asserted.
//
// This panel exists because "explainable AI" that only shows *an* explanation, without
// disclosing how faithful/calibrated/stable that explanation and the underlying model's
// confidence actually are, is not meaningfully more trustworthy than a black box -- see
// docs/TRUSTWORTHINESS_AUDIT.md's stated purpose.

interface TrustFinding {
  label: string;
  value: string;
  meaning: string;
  source: string;
}

const FINDINGS: TrustFinding[] = [
  {
    label: "Explanation faithfulness",
    value: "≈ 0 (not distinguishable from random)",
    meaning:
      "Removing the top attributed features degrades this model's prediction about as much as removing random features. Tested 4 independent ways (2 IG baselines, Gradient×Input, Occlusion) at sample sizes up to n=60 — all agree. Treat the “Model-attributed evidence” below as what the model responded to, not proof of the true reason.",
    source: "docs/XAI_EVALUATION.md §Deep diagnostic",
  },
  {
    label: "Predictive accuracy (financial-only, walk-forward)",
    value: "38–40% on a 3-class problem",
    meaning:
      "Statistically indistinguishable from a class-frequency baseline across 5 real out-of-sample folds — consistent with market efficiency. The confidence percentage above should be read against that backdrop, not as evidence of a reliably accurate model.",
    source: "docs/EXPERIMENT_PROTOCOL.md §Walk-forward results",
  },
  {
    label: "Confidence calibration",
    value: "No tested correction unambiguously improves it",
    meaning:
      "Global and per-regime temperature scaling were tested against real held-out data; global scaling improves one metric (ECE) but worsens two others (Brier, NLL). The confidence percentage shown is the model's raw softmax output, not a corrected probability.",
    source: "docs/EXPERIMENT_PROTOCOL.md §Calibration",
  },
  {
    label: "Regime-adaptive fusion (RAMA)",
    value: "Statistically real, practically small (≈ 1.6 pp range)",
    meaning:
      "The modality weights below do shift by regime — a permutation test confirms this is not chance (p<0.0002) — but the shift is small. Don't read a large qualitative story into small differences between regimes.",
    source: "docs/EXPERIMENT_PROTOCOL.md §Fusion-gate diagnostic",
  },
];

export function TrustworthinessPanel() {
  return (
    <details className="trust-panel">
      <summary>How much should I trust this? (real, measured findings)</summary>
      <div className="trust-findings">
        {FINDINGS.map((f) => (
          <div key={f.label} className="trust-finding">
            <div className="trust-finding-head">
              <span className="trust-label">{f.label}</span>
              <span className="trust-value">{f.value}</span>
            </div>
            <p className="trust-meaning">{f.meaning}</p>
            <p className="trust-source">Source: {f.source}</p>
          </div>
        ))}
      </div>
      <p className="trust-footer muted">
        These are the results of a full diagnostic audit of this system, not a legal
        disclaimer template — see <code>docs/TRUSTWORTHINESS_AUDIT.md</code> for the
        complete methodology, every root-cause investigation, and what was and wasn't
        fixable.
      </p>
    </details>
  );
}
