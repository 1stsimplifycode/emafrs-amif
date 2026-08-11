import { useStagedProgress } from "../../lib/useStagedProgress";
import "./StagedProgressIndicator.css";

interface Props {
  active: boolean;
}

// Shown while a real ~10-30s training/inference request is in flight. The label text
// names the REAL steps the backend actually runs, in their real order -- see
// useStagedProgress.ts for exactly what is and isn't simulated here (the sequence of
// steps is real; only the timing of when to advance the label is a UI-side estimate).
export function StagedProgressIndicator({ active }: Props) {
  const label = useStagedProgress(active);
  if (!active) return null;
  return (
    <div className="staged-progress" role="status" aria-live="polite">
      <span className="staged-progress-spinner" aria-hidden="true" />
      <span className="staged-progress-label">{label}</span>
    </div>
  );
}
