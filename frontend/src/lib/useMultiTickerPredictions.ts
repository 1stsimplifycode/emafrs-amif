import { useEffect, useState } from "react";
import { api, ApiError } from "../api";
import type { PredictionResponse } from "../api";
import { getCachedPrediction, setCachedPrediction } from "./predictionCache";

export type TickerPredictionState =
  | { status: "loading"; isRevalidating?: false }
  | { status: "ready"; prediction: PredictionResponse; isRevalidating: boolean }
  | { status: "not-computed" } // real 404 -- no prediction exists yet for this ticker, never fabricated
  | { status: "error"; message: string };

// Fetches the LATEST persisted prediction for each ticker in parallel -- real calls to
// the real backend, one per ticker. This never trains a fresh model (that is what
// Refresh does explicitly, on user action, since it costs ~10-30s); it only reads
// whatever has already been computed and persisted, which is why some tickers can
// legitimately come back "not-computed" rather than a fabricated placeholder value.
//
// Shares the same real-response cache as usePrediction (predictionCache.ts): a ticker
// already viewed on the Asset Profile page shows its real cached value here instantly
// too, revalidating silently in the background -- one cache, real data, no duplicate
// "loading" flash for something already fetched this session.
export function useMultiTickerPredictions(tickers: string[]) {
  const [state, setState] = useState<Record<string, TickerPredictionState>>({});

  useEffect(() => {
    let cancelled = false;
    setState((prev) => {
      const next = { ...prev };
      for (const t of tickers) {
        if (next[t]) continue;
        const cached = getCachedPrediction(t);
        next[t] = cached ? { status: "ready", prediction: cached, isRevalidating: true } : { status: "loading" };
      }
      return next;
    });

    tickers.forEach((ticker) => {
      api
        .latestPrediction(ticker)
        .then((prediction) => {
          if (cancelled) return;
          setCachedPrediction(ticker, prediction);
          setState((prev) => ({ ...prev, [ticker]: { status: "ready", prediction, isRevalidating: false } }));
        })
        .catch((e) => {
          if (cancelled) return;
          if (e instanceof ApiError && e.status === 404) {
            setState((prev) => ({ ...prev, [ticker]: { status: "not-computed" } }));
          } else {
            setState((prev) => ({ ...prev, [ticker]: { status: "error", message: String(e) } }));
          }
        });
    });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tickers.join(",")]);

  return state;
}
