import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { EvidenceQualityPanel } from "./EvidenceQualityPanel";

function renderPanel(
  weights: Record<string, number>,
  confidence = 0.42,
  signalConsistency: number | null = null
) {
  return render(
    <MemoryRouter>
      <EvidenceQualityPanel confidence={confidence} modalityWeights={weights} signalConsistency={signalConsistency} />
    </MemoryRouter>
  );
}

describe("EvidenceQualityPanel", () => {
  it("renders the real confidence value, not a fabricated trust score", () => {
    renderPanel({ financial: 1.0 }, 0.42);
    expect(screen.getByText("42%")).toBeInTheDocument();
  });

  it("marks present modalities as available and shows the availability ratio", () => {
    renderPanel({ financial: 0.5, microstructure: 0.5 });
    expect(screen.getByText(/2 of 3 known evidence sources/)).toBeInTheDocument();
    expect(screen.getByText("✓ Financial")).toBeInTheDocument();
    expect(screen.getByText("✓ Microstructure")).toBeInTheDocument();
    expect(screen.getByText("— Text/affective")).toBeInTheDocument();
  });

  it("renders the real, live-computed consistency value when the backend provides one", () => {
    renderPanel({ financial: 0.5, microstructure: 0.5 }, 0.42, 0.73);
    expect(screen.getByText("Agree")).toBeInTheDocument();
    expect(screen.getByText("0.73")).toBeInTheDocument();
  });

  it("renders a conflict pill for a real negative consistency value", () => {
    renderPanel({ financial: 0.5, microstructure: 0.5 }, 0.42, -0.5);
    expect(screen.getByText("Conflict")).toBeInTheDocument();
    expect(screen.getByText("-0.50")).toBeInTheDocument();
  });

  it("renders an ambiguous pill for an exact-zero real consistency value", () => {
    renderPanel({ financial: 0.5, microstructure: 0.5 }, 0.42, 0.0);
    expect(screen.getByText("Ambiguous")).toBeInTheDocument();
  });

  it("never fabricates a consistency score when null -- states plainly why it's undefined", () => {
    renderPanel({ financial: 1.0 }, 0.42, null);
    expect(screen.getByText(/Not computable for this prediction/)).toBeInTheDocument();
    expect(screen.getByText(/only 1 evidence source contributed/)).toBeInTheDocument();
  });

  it("lists missing modalities with a reason, when any are absent", () => {
    renderPanel({ financial: 1.0 });
    expect(screen.getByText("What's missing")).toBeInTheDocument();
    expect(screen.getByText(/Microstructure:/)).toBeInTheDocument();
    expect(screen.getByText(/Text\/affective:/)).toBeInTheDocument();
  });

  it("omits the missing-evidence section when every known modality is present", () => {
    renderPanel({ financial: 0.4, microstructure: 0.3, text: 0.3 });
    expect(screen.queryByText("What's missing")).not.toBeInTheDocument();
  });
});
