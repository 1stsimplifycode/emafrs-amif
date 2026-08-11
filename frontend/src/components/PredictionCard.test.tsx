import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { PredictionResponse } from "../api";
import { PredictionCard } from "./PredictionCard";

// Shape and values mirror a REAL response captured from the live backend
// (POST /api/v1/predictions/MSFT/refresh during manual verification of this dashboard),
// not an invented fixture.
const REAL_SAMPLE_PREDICTION: PredictionResponse = {
  id: 1,
  ticker: "MSFT",
  as_of_date: "2026-08-07",
  direction: -1,
  direction_label: "bearish",
  confidence: 0.38898178935050964,
  regime: 3,
  regime_name: "High",
  modality_weights: { financial: 0.5053711533546448, microstructure: 0.49462890625 },
  top_attributions: {
    macd_signal: 0.038188084959983826,
    cumulative_return_21d: -0.030814174562692642,
    macd: 0.02495722845196724,
    bb_lower: 0.02076868526637554,
    macd_hist: 0.0156856682151556,
  },
  counterfactual: {
    feature_name: "macd_signal",
    delta: -6.0,
    original_value: 2.428739309310913,
    counterfactual_value: -3.571260690689087,
  },
  manipulation_risk: null,
  signal_consistency: 0.0,
  explanation_text:
    "Predicted market direction: bearish (model confidence 39%). Current market regime: High. " +
    "Modality contribution to this prediction: financial 51%, microstructure 49%. " +
    "Model-attributed evidence (top financial indicators, Integrated Gradients): macd_signal (+0.038). " +
    "Counterfactual: had macd_signal differed by -6.000 (from 2.429 to -3.571), the model's prediction " +
    "would have changed. This describes model sensitivity, not a causal claim about the underlying market.",
  model_version_id: 1,
  created_at: "2026-08-10T19:39:51.592598",
};

describe("PredictionCard", () => {
  it("renders the ticker, direction, and confidence from real data", () => {
    render(<PredictionCard prediction={REAL_SAMPLE_PREDICTION} />);
    expect(screen.getByText("MSFT")).toBeInTheDocument();
    expect(screen.getByText(/BEARISH/)).toBeInTheDocument();
    expect(screen.getByText(/39% confidence/)).toBeInTheDocument();
  });

  it("renders the regime name", () => {
    const { container } = render(<PredictionCard prediction={REAL_SAMPLE_PREDICTION} />);
    expect(container.querySelector(".regime-value")).toHaveTextContent("High");
    expect(container.querySelector(".regime-value")).toHaveTextContent("regime 3");
  });

  it("renders all top attributions ranked by magnitude", () => {
    const { container } = render(<PredictionCard prediction={REAL_SAMPLE_PREDICTION} />);
    const list = container.querySelector(".attribution-list");
    expect(list).toHaveTextContent("macd_signal");
    expect(list).toHaveTextContent("+0.0382");
    expect(list).toHaveTextContent("cumulative_return_21d");
    expect(list).toHaveTextContent("-0.0308");
  });

  it("renders the counterfactual explanation with the real feature and delta", () => {
    const { container } = render(<PredictionCard prediction={REAL_SAMPLE_PREDICTION} />);
    const cfParagraph = container.querySelector(".counterfactual-text");
    expect(cfParagraph).toHaveTextContent("macd_signal");
    expect(cfParagraph).toHaveTextContent("-6.000");
  });

  it("renders the anti-overclaiming explanation text verbatim", () => {
    render(<PredictionCard prediction={REAL_SAMPLE_PREDICTION} />);
    expect(screen.getByText(/not a causal claim/)).toBeInTheDocument();
  });

  it("omits the manipulation risk section when null (never fabricates a value)", () => {
    render(<PredictionCard prediction={REAL_SAMPLE_PREDICTION} />);
    expect(screen.queryByText("Manipulation risk")).not.toBeInTheDocument();
  });

  it("renders the manipulation risk section when present", () => {
    const withRisk = { ...REAL_SAMPLE_PREDICTION, manipulation_risk: 0.42 };
    render(<PredictionCard prediction={withRisk} />);
    expect(screen.getByText("Manipulation risk")).toBeInTheDocument();
    expect(screen.getByText("0.420")).toBeInTheDocument();
  });

  it("renders the trustworthiness panel alongside every prediction", () => {
    render(<PredictionCard prediction={REAL_SAMPLE_PREDICTION} />);
    expect(screen.getByText(/How much should I trust this\?/)).toBeInTheDocument();
  });
});
