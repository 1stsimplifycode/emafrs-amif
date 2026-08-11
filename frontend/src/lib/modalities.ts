// Canonical modality metadata shared by every evidence visualization -- one source of
// truth so the Consumer and Research experiences describe the same modality the same way.
export interface ModalityMeta {
  label: string;
  short: string;
  color: string;
  description: string;
}

export const MODALITY_META: Record<string, ModalityMeta> = {
  financial: {
    label: "Financial signals",
    short: "Financial",
    color: "var(--signal)",
    description: "Price/technical indicators (momentum, volatility, trend).",
  },
  microstructure: {
    label: "Market behavior",
    short: "Microstructure",
    color: "var(--structural)",
    description: "Order-flow/liquidity proxies from recent intraday trading.",
  },
  text: {
    label: "News & affective context",
    short: "Text/affective",
    color: "var(--context)",
    description: "Sentiment and affective signal extracted from financial text.",
  },
  emotion: {
    label: "Affective context",
    short: "Affective",
    color: "var(--context)",
    description: "Affective/emotion signal extracted from financial text.",
  },
};

export function modalityMeta(name: string): ModalityMeta {
  return (
    MODALITY_META[name] ?? {
      label: name,
      short: name,
      color: "var(--neutral)",
      description: "",
    }
  );
}

// The modalities EMAFRS's research taxonomy recognizes, whether or not they appear in
// any given live prediction -- used to honestly show what's MISSING, not just what's
// present, per the product's "what evidence is available" requirement.
export const ALL_KNOWN_MODALITIES = ["financial", "microstructure", "text"];
