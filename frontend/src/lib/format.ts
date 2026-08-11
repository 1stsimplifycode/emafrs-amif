export function directionPillClass(label: string): string {
  if (label === "bullish") return "pill pill-bullish";
  if (label === "bearish") return "pill pill-bearish";
  return "pill pill-neutral";
}

export function directionWord(label: string): string {
  return label.charAt(0).toUpperCase() + label.slice(1);
}

export function pct(x: number): string {
  return `${(x * 100).toFixed(0)}%`;
}

export function pct1(x: number): string {
  return `${(x * 100).toFixed(1)}%`;
}

export function signed(x: number, digits = 4): string {
  return `${x >= 0 ? "+" : ""}${x.toFixed(digits)}`;
}
