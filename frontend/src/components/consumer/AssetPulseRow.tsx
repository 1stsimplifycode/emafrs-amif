import { Link } from "react-router-dom";
import type { TickerPredictionState } from "../../lib/useMultiTickerPredictions";
import { directionPillClass, directionWord, pct } from "../../lib/format";
import { friendlyErrorMessage } from "../../lib/errors";

interface Props {
  ticker: string;
  state: TickerPredictionState;
  onRemove?: () => void;
}

export function AssetPulseRow({ ticker, state, onRemove }: Props) {
  return (
    <Link to={`/asset/${ticker}`} className="pulse-row">
      <span className="pulse-ticker">{ticker}</span>

      {state.status === "loading" && <span className="skeleton pulse-skeleton" />}

      {state.status === "not-computed" && (
        <span className="text-faint pulse-status">No prediction yet — open to compute one</span>
      )}

      {state.status === "error" && (
        <span className="error-banner pulse-status">{friendlyErrorMessage(`${ticker}'s outlook`)}</span>
      )}

      {state.status === "ready" && (
        <>
          <span className={directionPillClass(state.prediction.direction_label)}>
            {directionWord(state.prediction.direction_label)}
          </span>
          <span className="pulse-confidence numeric">{pct(state.prediction.confidence)}</span>
          <span className="text-faint pulse-regime">
            {state.prediction.regime_name} regime
            {state.isRevalidating && <span className="revalidating-dot" title="Confirming with a live fetch…" />}
          </span>
        </>
      )}

      {onRemove && (
        <button
          className="btn btn-ghost pulse-remove"
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onRemove();
          }}
          aria-label={`Remove ${ticker} from watchlist`}
        >
          ×
        </button>
      )}
    </Link>
  );
}
