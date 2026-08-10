import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "./api";

function mockFetchOnce(status: number, body: unknown) {
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    statusText: "mock",
    json: async () => body,
  }) as unknown as typeof fetch;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("api.health", () => {
  it("returns the parsed health payload on success", async () => {
    mockFetchOnce(200, { status: "ok", version: "0.1.0" });
    const result = await api.health();
    expect(result).toEqual({ status: "ok", version: "0.1.0" });
  });
});

describe("api error handling", () => {
  it("throws ApiError with the backend's detail message on non-2xx", async () => {
    mockFetchOnce(404, { detail: "No prediction has been computed yet for ZZZZ." });
    await expect(api.latestPrediction("ZZZZ")).rejects.toThrow(
      "No prediction has been computed yet for ZZZZ."
    );
  });

  it("ApiError carries the HTTP status code", async () => {
    mockFetchOnce(429, { detail: "Rate limit exceeded" });
    try {
      await api.latestPrediction("ZZZZ");
      expect.fail("expected ApiError to be thrown");
    } catch (e) {
      expect(e).toBeInstanceOf(ApiError);
      expect((e as ApiError).status).toBe(429);
    }
  });

  it("falls back to statusText when the error body is not JSON", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      json: async () => {
        throw new Error("not json");
      },
    }) as unknown as typeof fetch;
    await expect(api.health()).rejects.toThrow("Internal Server Error");
  });
});

describe("api.refreshPrediction", () => {
  it("POSTs to the refresh endpoint", async () => {
    mockFetchOnce(200, { id: 1, ticker: "AAPL" });
    await api.refreshPrediction("AAPL");
    expect(globalThis.fetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/v1/predictions/AAPL/refresh"),
      expect.objectContaining({ method: "POST" })
    );
  });
});
