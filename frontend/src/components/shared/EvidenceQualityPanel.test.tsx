import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { EvidenceQualityPanel } from "./EvidenceQualityPanel";

function renderPanel(weights: Record<string, number>, confidence = 0.42) {
  return render(
    <MemoryRouter>
      <EvidenceQualityPanel confidence={confidence} modalityWeights={weights} />
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

  it("never fabricates a consistency score -- states it plainly as not computed", () => {
    renderPanel({ financial: 1.0 });
    expect(screen.getByText(/Not currently computed for live predictions/)).toBeInTheDocument();
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
