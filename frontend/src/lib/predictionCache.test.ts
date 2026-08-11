import { beforeEach, describe, expect, it } from "vitest";
import type { PredictionResponse } from "../api";
import { clearPredictionCache, getCachedPrediction, setCachedPrediction } from "./predictionCache";

function fakePrediction(overrides: Partial<PredictionResponse> = {}): PredictionResponse {
  return {
    id: 1, ticker: "SPY", as_of_date: "2026-08-10", direction: 1, direction_label: "bullish",
    confidence: 0.5, regime: 1, regime_name: "Low",
    modality_weights: { financial: 1.0 }, top_attributions: {}, counterfactual: null,
    manipulation_risk: null, signal_consistency: null,
    explanation_text: "x", model_version_id: 1, created_at: "2026-08-10T00:00:00Z",
    ...overrides,
  };
}

describe("predictionCache", () => {
  beforeEach(() => {
    clearPredictionCache();
  });

  it("returns undefined for a ticker never cached", () => {
    expect(getCachedPrediction("NEVER_CACHED")).toBeUndefined();
  });

  it("returns the exact real value that was set, unmodified", () => {
    const p = fakePrediction({ ticker: "MSFT", confidence: 0.77 });
    setCachedPrediction("MSFT", p);
    expect(getCachedPrediction("MSFT")).toBe(p);
  });

  it("overwrites a stale cached value with a newer real one", () => {
    setCachedPrediction("SPY", fakePrediction({ id: 1 }));
    setCachedPrediction("SPY", fakePrediction({ id: 2 }));
    expect(getCachedPrediction("SPY")?.id).toBe(2);
  });

  it("keeps separate entries per ticker", () => {
    setCachedPrediction("SPY", fakePrediction({ ticker: "SPY" }));
    setCachedPrediction("QQQ", fakePrediction({ ticker: "QQQ" }));
    expect(getCachedPrediction("SPY")?.ticker).toBe("SPY");
    expect(getCachedPrediction("QQQ")?.ticker).toBe("QQQ");
  });
});
