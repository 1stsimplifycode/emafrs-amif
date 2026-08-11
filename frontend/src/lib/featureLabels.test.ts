import { describe, expect, it } from "vitest";
import { featureLabel } from "./featureLabels";

describe("featureLabel", () => {
  it("maps known real backend feature names to readable labels", () => {
    expect(featureLabel("macd_signal")).toBe("MACD Signal Line");
    expect(featureLabel("cumulative_return_21d")).toBe("21-Day Cumulative Return");
    expect(featureLabel("rsi_14")).toBe("14-Day RSI");
  });

  it("falls back to Title Case for an unmapped feature rather than inventing a description", () => {
    expect(featureLabel("some_future_feature")).toBe("Some Future Feature");
  });
});
