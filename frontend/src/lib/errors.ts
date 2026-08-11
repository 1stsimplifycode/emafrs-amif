import { ApiError } from "../api";

// Consumer-facing error copy -- honest that something failed, without surfacing raw
// exception text ("TypeError: Failed to fetch", HTTP status internals, etc.) to a
// non-technical audience. Research pages are allowed to show the raw message (they're
// meant to be technical); this helper is for the Consumer product only.
export function friendlyErrorMessage(context: string): string {
  return `Something went wrong loading ${context} right now. Please try again in a moment.`;
}

export function isNotFound(e: unknown): boolean {
  return e instanceof ApiError && e.status === 404;
}
