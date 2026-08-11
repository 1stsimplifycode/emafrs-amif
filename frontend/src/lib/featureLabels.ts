// Human-readable labels for the REAL financial feature columns the backend produces
// (src/emafrs/features/technical.py, src/emafrs/models/dataset.py) -- this maps the same
// real feature/value pairs the API returns to consumer-friendly names; it never changes,
// invents, or drops a value. Any feature name not in this table (e.g. a future addition)
// falls back to a lightly formatted version of the raw name rather than a made-up label.
const FEATURE_LABELS: Record<string, string> = {
  return_1d: "1-Day Return",
  cumulative_return_21d: "21-Day Cumulative Return",
  realized_vol_21d: "21-Day Realized Volatility",
  max_drawdown_252d: "252-Day Max Drawdown",
  sma_10: "10-Day Moving Average",
  sma_50: "50-Day Moving Average",
  ema_12: "12-Day Exponential Moving Average",
  ema_26: "26-Day Exponential Moving Average",
  rsi_14: "14-Day RSI",
  volume_zscore_21d: "Volume (21-Day Z-Score)",
  macd: "MACD",
  macd_signal: "MACD Signal Line",
  macd_hist: "MACD Histogram",
  bb_mid: "Bollinger Band (Midline)",
  bb_upper: "Bollinger Band (Upper)",
  bb_lower: "Bollinger Band (Lower)",
  bb_pct_b: "Bollinger %B",
  var_95: "Value at Risk (95%)",
  cvar_95: "Conditional VaR (95%)",
};

export function featureLabel(name: string): string {
  if (FEATURE_LABELS[name]) return FEATURE_LABELS[name];
  // Fallback for any feature not yet in the table above: snake_case -> Title Case,
  // never a fabricated description of what the feature means.
  return name
    .split("_")
    .map((w) => (w.length ? w[0].toUpperCase() + w.slice(1) : w))
    .join(" ");
}
