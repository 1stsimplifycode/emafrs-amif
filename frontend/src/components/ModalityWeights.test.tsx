import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ModalityWeights } from "./ModalityWeights";

describe("ModalityWeights", () => {
  it("renders a row per modality with its real percentage", () => {
    render(<ModalityWeights weights={{ financial: 0.505, microstructure: 0.495 }} />);
    expect(screen.getByText("financial")).toBeInTheDocument();
    expect(screen.getByText("microstructure")).toBeInTheDocument();
    expect(screen.getByText("50.5%")).toBeInTheDocument();
    expect(screen.getByText("49.5%")).toBeInTheDocument();
  });

  it("sorts modalities by weight descending", () => {
    render(<ModalityWeights weights={{ microstructure: 0.3, financial: 0.7 }} />);
    const labels = screen.getAllByText(/financial|microstructure/).map((el) => el.textContent);
    expect(labels[0]).toBe("financial"); // higher weight (0.7) rendered first
  });

  it("shows a fallback message when no weights are present", () => {
    render(<ModalityWeights weights={{}} />);
    expect(screen.getByText(/No modality weights available/)).toBeInTheDocument();
  });

  it("bar width reflects the real weight proportion, not a fixed value", () => {
    const { container } = render(<ModalityWeights weights={{ financial: 0.8, microstructure: 0.2 }} />);
    const fills = container.querySelectorAll(".modality-bar-fill");
    expect(fills[0]).toHaveStyle({ width: "80%" });
    expect(fills[1]).toHaveStyle({ width: "20%" });
  });
});
