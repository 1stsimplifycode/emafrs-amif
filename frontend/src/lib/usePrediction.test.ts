import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api";
import type { PredictionResponse } from "../api";
import { clearPredictionCache, setCachedPrediction } from "./predictionCache";
import { usePrediction } from "./usePrediction";

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return { ...actual, api: { ...actual.api, latestPrediction: vi.fn(), refreshPrediction: vi.fn() } };
});

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

beforeEach(() => {
  clearPredictionCache();
  vi.mocked(api.latestPrediction).mockReset();
  vi.mocked(api.refreshPrediction).mockReset();
});

describe("usePrediction", () => {
  it("starts in loading state with no cache, then shows the real fetched value", async () => {
    const real = fakePrediction({ id: 42 });
    vi.mocked(api.latestPrediction).mockResolvedValue(real);

    const { result } = renderHook(() => usePrediction("SPY"));
    expect(result.current.status).toBe("loading");

    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current.prediction?.id).toBe(42);
    expect(result.current.isRevalidating).toBe(false);
  });

  it("shows the cached real value INSTANTLY (not loading) when one exists, then silently revalidates", async () => {
    const cached = fakePrediction({ id: 1, confidence: 0.3 });
    setCachedPrediction("SPY", cached);
    const fresh = fakePrediction({ id: 1, confidence: 0.9 });
    vi.mocked(api.latestPrediction).mockResolvedValue(fresh);

    const { result } = renderHook(() => usePrediction("SPY"));
    // instantly ready from cache, not "loading" -- the core stale-while-revalidate behavior
    expect(result.current.status).toBe("ready");
    expect(result.current.prediction?.confidence).toBe(0.3);
    expect(result.current.isRevalidating).toBe(true);

    await waitFor(() => expect(result.current.isRevalidating).toBe(false));
    expect(result.current.prediction?.confidence).toBe(0.9); // replaced by the real fresh value
  });

  it("reflects a real 404 as not-computed, never a fabricated prediction", async () => {
    const { ApiError } = await import("../api");
    vi.mocked(api.latestPrediction).mockRejectedValue(new ApiError(404, "not found"));

    const { result } = renderHook(() => usePrediction("NEVERCOMPUTED"));
    await waitFor(() => expect(result.current.status).toBe("not-computed"));
    expect(result.current.prediction).toBeNull();
  });

  it("refresh() calls the real refresh endpoint and updates both state and cache", async () => {
    vi.mocked(api.latestPrediction).mockRejectedValue(
      new (await import("../api")).ApiError(404, "not found")
    );
    const refreshed = fakePrediction({ id: 99, confidence: 0.65 });
    vi.mocked(api.refreshPrediction).mockResolvedValue(refreshed);

    const { result } = renderHook(() => usePrediction("SPY"));
    await waitFor(() => expect(result.current.status).toBe("not-computed"));

    await result.current.refresh();
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current.prediction?.id).toBe(99);
  });
});
