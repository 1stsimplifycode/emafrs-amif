import type { PredictionResponse } from "../api";

// A simple in-memory, session-scoped cache of real API responses -- never fabricated
// data. Its only purpose is stale-while-revalidate: when you've already fetched a
// ticker's real prediction once, later visits show that real (possibly slightly stale)
// value instantly instead of a loading spinner, while a fresh real fetch happens
// silently in the background and replaces it the moment it lands. Resets on a full page
// reload (module-level state, not persisted) -- it is a perceived-speed optimization,
// not a data store.
const cache = new Map<string, PredictionResponse>();

export function getCachedPrediction(ticker: string): PredictionResponse | undefined {
  return cache.get(ticker);
}

export function setCachedPrediction(ticker: string, prediction: PredictionResponse): void {
  cache.set(ticker, prediction);
}

export function clearPredictionCache(): void {
  cache.clear();
}
