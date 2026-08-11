import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EvidenceConstellation } from "./EvidenceConstellation";

describe("EvidenceConstellation", () => {
  it("renders a node with the real weight percentage for each present modality", () => {
    const { container } = render(
      <EvidenceConstellation
        weights={{ financial: 0.6, microstructure: 0.4 }}
        outlookLabel="Bullish"
        outlookColorVar="var(--bullish)"
      />
    );
    const text = container.textContent ?? "";
    expect(text).toContain("60%");
    expect(text).toContain("40%");
    expect(text).toContain("Financial");
    expect(text).toContain("Microstructure");
  });

  it("renders a ghost node for a known modality that produced no evidence", () => {
    const { container } = render(
      <EvidenceConstellation weights={{ financial: 1.0 }} outlookLabel="Bearish" outlookColorVar="var(--bearish)" />
    );
    const text = container.textContent ?? "";
    expect(text).toContain("no evidence");
    expect(text).toContain("Text/affective");
  });

  it("renders the real outlook label at the center", () => {
    const { container } = render(
      <EvidenceConstellation weights={{ financial: 1.0 }} outlookLabel="Neutral" outlookColorVar="var(--neutral)" />
    );
    expect(container.textContent).toContain("Neutral");
    expect(container.textContent).toContain("Outlook");
  });
});
