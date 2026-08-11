import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TRAINING_STAGES, useStagedProgress } from "./useStagedProgress";

describe("useStagedProgress", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("starts at the first real stage label when active", () => {
    const { result } = renderHook(() => useStagedProgress(true, 1000));
    expect(result.current).toBe(TRAINING_STAGES[0]);
  });

  it("advances through the real stage labels over time, in order", () => {
    const { result } = renderHook(() => useStagedProgress(true, 1000));
    act(() => vi.advanceTimersByTime(1000));
    expect(result.current).toBe(TRAINING_STAGES[1]);
    act(() => vi.advanceTimersByTime(1000));
    expect(result.current).toBe(TRAINING_STAGES[2]);
  });

  it("holds on the last stage rather than looping back to the first", () => {
    const { result } = renderHook(() => useStagedProgress(true, 1000));
    act(() => vi.advanceTimersByTime(1000 * (TRAINING_STAGES.length + 5)));
    expect(result.current).toBe(TRAINING_STAGES[TRAINING_STAGES.length - 1]);
  });

  it("resets to the first stage when inactive", () => {
    const { result, rerender } = renderHook(({ active }) => useStagedProgress(active, 1000), {
      initialProps: { active: true },
    });
    act(() => vi.advanceTimersByTime(2000));
    expect(result.current).not.toBe(TRAINING_STAGES[0]);

    rerender({ active: false });
    expect(result.current).toBe(TRAINING_STAGES[0]);
  });
});
