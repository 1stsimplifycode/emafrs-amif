import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { SentimentPage } from "./SentimentPage";
import { api } from "../../api";
import type { PredictionResponse } from "../../api";

vi.mock("../../api", async () => {
  const actual = await vi.importActual<typeof import("../../api")>("../../api");
  return { ...actual, api: { ...actual.api, latestPrediction: vi.fn() } };
});

function basePrediction(overrides: Partial<PredictionResponse> = {}): PredictionResponse {
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
  window.localStorage.clear();
  window.localStorage.setItem("emafrs.watchlist.v1", JSON.stringify(["SPY"]));
  vi.mocked(api.latestPrediction).mockReset();
});

describe("SentimentPage live check (not a static claim)", () => {
  it("says not-yet-checked when no tracked ticker has a real prediction computed", async () => {
    const { ApiError } = await import("../../api");
    vi.mocked(api.latestPrediction).mockRejectedValue(new ApiError(404, "not found"));

    render(
      <MemoryRouter>
        <SentimentPage />
      </MemoryRouter>
    );

    await waitFor(() => expect(screen.getByText(/Not yet checked/)).toBeInTheDocument());
  });

  it("says yes, live-checked, when a real loaded prediction includes text evidence", async () => {
    vi.mocked(api.latestPrediction).mockResolvedValue(
      basePrediction({ modality_weights: { financial: 0.5, text: 0.5 } })
    );

    render(
      <MemoryRouter>
        <SentimentPage />
      </MemoryRouter>
    );

    await waitFor(() => expect(screen.getByText("Yes")).toBeInTheDocument());
  });

  it("says not currently, live-checked, when real loaded predictions never include text", async () => {
    vi.mocked(api.latestPrediction).mockResolvedValue(
      basePrediction({ modality_weights: { financial: 0.5, microstructure: 0.5 } })
    );

    render(
      <MemoryRouter>
        <SentimentPage />
      </MemoryRouter>
    );

    await waitFor(() => expect(screen.getByText("Not currently")).toBeInTheDocument());
    expect(screen.getByText(/checked live across all/)).toBeInTheDocument();
  });
});
