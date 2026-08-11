import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TrustworthinessPanel } from "./TrustworthinessPanel";

describe("TrustworthinessPanel", () => {
  it("renders collapsed by default (details without open attribute)", () => {
    const { container } = render(<TrustworthinessPanel />);
    const details = container.querySelector("details.trust-panel");
    expect(details).toBeInTheDocument();
    expect(details).not.toHaveAttribute("open");
  });

  it("renders the real faithfulness finding with its source citation", () => {
    render(<TrustworthinessPanel />);
    expect(screen.getByText("Explanation faithfulness")).toBeInTheDocument();
    expect(screen.getByText(/not distinguishable from random/)).toBeInTheDocument();
    expect(screen.getByText(/docs\/XAI_EVALUATION\.md/)).toBeInTheDocument();
  });

  it("renders the real calibration finding, not a rounded-up claim", () => {
    render(<TrustworthinessPanel />);
    expect(screen.getByText("Confidence calibration")).toBeInTheDocument();
    expect(screen.getByText(/No tested correction unambiguously improves it/)).toBeInTheDocument();
  });

  it("renders the RAMA regime-adaptivity finding with the real effect size", () => {
    render(<TrustworthinessPanel />);
    expect(screen.getByText("Regime-adaptive fusion (RAMA)")).toBeInTheDocument();
    expect(screen.getByText(/1\.6 pp range/)).toBeInTheDocument();
  });

  it("renders a source citation for every finding", () => {
    const { container } = render(<TrustworthinessPanel />);
    const sources = container.querySelectorAll(".trust-source");
    expect(sources.length).toBeGreaterThanOrEqual(4);
    sources.forEach((s) => expect(s.textContent).toMatch(/^Source: docs\//));
  });
});
