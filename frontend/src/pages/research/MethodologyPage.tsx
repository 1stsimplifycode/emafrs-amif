import "../../styles/research.css";

export function MethodologyPage() {
  return (
    <div>
      <div className="research-page-head">
        <span className="eyebrow">Methodology</span>
        <h1 className="display-xl">How these findings were actually verified</h1>
      </div>

      <div className="card finding-card">
        <span className="display-md">No-fabrication rule</span>
        <p>
          Every claim on every Research page is either backed by a real run cited to a
          specific script or document, or marked <strong>RESULT NOT AVAILABLE</strong> /{" "}
          <strong>NOT EVALUATED</strong> with the specific missing capability named.
          Nothing is estimated, interpolated, or presented as plausible-sounding filler.
          Where a correction was attempted and did not work, that is reported as the
          finding.
        </p>
      </div>

      <div className="card finding-card">
        <span className="display-md">Leakage auditing — truncation proofs</span>
        <p>
          Every rolling feature, the regime-detection engine, and the text/news
          date-mapping logic are tested the same way: recompute the value on a{" "}
          <em>truncated</em> history and compare to the value on the full series. If a
          computation secretly depends on future rows, the truncated and full values
          diverge and the test fails. Two deliberately-leaky feature functions are kept
          in the test suite specifically to prove this check has teeth, not just to
          report "pass" trivially.
        </p>
      </div>

      <div className="card finding-card">
        <span className="display-md">Walk-forward validation</span>
        <p>
          Where the data supports it, models are evaluated on expanding-window,
          non-overlapping folds — training data only ever grows forward in time, test
          windows never repeat or go backward. Where this isn't possible (e.g.
          microstructure's short real intraday-data window), that limitation is
          disclosed explicitly rather than silently evaluated on a single split and
          presented as equivalent.
        </p>
      </div>

      <div className="card finding-card">
        <span className="display-md">Statistical significance over eyeballing</span>
        <p>
          Findings that could plausibly be "probably nothing" or "probably real" are
          tested, not asserted — paired Wilcoxon/t-tests across folds, and a 5,000-shuffle
          permutation test for RAMA's regime-adaptivity, rather than declaring a pattern
          significant because it looks that way in a table.
        </p>
      </div>

      <div className="card finding-card">
        <span className="display-md">The remediation loop</span>
        <p>
          For every weak or failed finding: diagnose the root cause with a real
          experiment (not a guess), implement a targeted correction where the data and
          codebase genuinely permit it, re-run for real, and report the before/after —
          including when the correction didn't work. Two corrections were tested and
          rejected on record in Regime Analysis; that's reported as plainly as the one
          that worked (regime-probability leakage).
        </p>
      </div>

      <p className="source-link">
        Source: docs/DATA_LEAKAGE_AUDIT.md, docs/EXPERIMENT_PROTOCOL.md, docs/TRUSTWORTHINESS_AUDIT.md
      </p>
    </div>
  );
}
