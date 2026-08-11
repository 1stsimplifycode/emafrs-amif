import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { useWatchlist } from "./watchlist";

describe("useWatchlist", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("starts with the default tracked tickers when nothing is stored", () => {
    const { result } = renderHook(() => useWatchlist());
    expect(result.current.tickers).toEqual(["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]);
  });

  it("adds a ticker, uppercased, without duplicating it", () => {
    const { result } = renderHook(() => useWatchlist());
    act(() => result.current.add("tsla"));
    expect(result.current.tickers).toContain("TSLA");
    act(() => result.current.add("TSLA"));
    expect(result.current.tickers.filter((t) => t === "TSLA")).toHaveLength(1);
  });

  it("removes a ticker", () => {
    const { result } = renderHook(() => useWatchlist());
    act(() => result.current.remove("SPY"));
    expect(result.current.tickers).not.toContain("SPY");
  });

  it("persists across hook instances via localStorage", () => {
    const { result, unmount } = renderHook(() => useWatchlist());
    act(() => result.current.add("META"));
    unmount();

    const { result: second } = renderHook(() => useWatchlist());
    expect(second.current.tickers).toContain("META");
  });

  it("falls back to defaults if localStorage contains corrupted data", () => {
    window.localStorage.setItem("emafrs.watchlist.v1", "{not valid json");
    const { result } = renderHook(() => useWatchlist());
    expect(result.current.tickers).toEqual(["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]);
  });
});
