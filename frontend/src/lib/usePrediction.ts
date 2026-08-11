import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api";
import type { PredictionResponse } from "../api";
import { friendlyErrorMessage } from "./errors";
import { getCachedPrediction, setCachedPrediction } from "./predictionCache";

export type PredictionHookState =
  | { status: "loading"; prediction: null; isRevalidating: false }
  | { status: "ready"; prediction: PredictionResponse; isRevalidating: boolean }
  | { status: "not-computed"; prediction: null; isRevalidating: false }
  | { status: "error"; prediction: null; message: string; isRevalidating: false };

// Shared real-data-fetching hook for a single ticker's latest prediction, used by both
// the Asset Profile and AI Intelligence pages. Stale-while-revalidate: if a real
// prediction for this ticker is already cached from an earlier visit this session, it's
// shown immediately (status "ready", isRevalidating true) while a fresh real fetch runs
// in the background and silently replaces it -- nothing here is fabricated, it's the
// same real API response, just shown without waiting for the network twice.
export function usePrediction(ticker: string) {
  const [state, setState] = useState<PredictionHookState>(() => {
    const cached = getCachedPrediction(ticker);
    return cached
      ? { status: "ready", prediction: cached, isRevalidating: true }
      : { status: "loading", prediction: null, isRevalidating: false };
  });

  useEffect(() => {
    let cancelled = false;
    const cached = getCachedPrediction(ticker);
    setState(
      cached
        ? { status: "ready", prediction: cached, isRevalidating: true }
        : { status: "loading", prediction: null, isRevalidating: false }
    );

    api
      .latestPrediction(ticker)
      .then((p) => {
        if (cancelled) return;
        setCachedPrediction(ticker, p);
        setState({ status: "ready", prediction: p, isRevalidating: false });
      })
      .catch((e) => {
        if (cancelled) return;
        if (e instanceof ApiError && e.status === 404) {
          setState({ status: "not-computed", prediction: null, isRevalidating: false });
        } else {
          setState({ status: "error", prediction: null, message: friendlyErrorMessage(`${ticker}'s prediction`), isRevalidating: false });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [ticker]);

  const refresh = useCallback(async () => {
    const p = await api.refreshPrediction(ticker);
    setCachedPrediction(ticker, p);
    setState({ status: "ready", prediction: p, isRevalidating: false });
    return p;
  }, [ticker]);

  return { ...state, refresh };
}
