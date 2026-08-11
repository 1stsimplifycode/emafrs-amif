import { useCallback, useEffect, useState } from "react";

// The backend has no user-account/watchlist concept -- this is a purely client-side,
// per-browser list of ticker symbols. It does not fabricate any market data: every
// ticker in the list is still resolved through the real /predictions endpoints. This is
// deliberately honest about what it is (a local list of symbols you're following), not
// dressed up as a synced/server-backed feature it isn't.
const STORAGE_KEY = "emafrs.watchlist.v1";
const DEFAULT_WATCHLIST = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"];

function readStored(): string[] {
  if (typeof window === "undefined") return DEFAULT_WATCHLIST;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_WATCHLIST;
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed) && parsed.every((t) => typeof t === "string")) {
      return parsed;
    }
  } catch {
    // corrupted localStorage value -- fall back to defaults rather than throwing
  }
  return DEFAULT_WATCHLIST;
}

export function useWatchlist() {
  const [tickers, setTickers] = useState<string[]>(() => readStored());

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(tickers));
  }, [tickers]);

  const add = useCallback((ticker: string) => {
    const t = ticker.trim().toUpperCase();
    if (!t) return;
    setTickers((prev) => (prev.includes(t) ? prev : [...prev, t]));
  }, []);

  const remove = useCallback((ticker: string) => {
    setTickers((prev) => prev.filter((t) => t !== ticker));
  }, []);

  return { tickers, add, remove };
}
