// Typed client for the EMAFRS FastAPI backend (src/emafrs/api/app.py).
// Every function here calls a real endpoint -- no mocked/canned data.

export const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export interface DataSourceHealth {
  provider: string;
  configured: boolean;
  reachable: boolean | null;
  last_success_at: string | null;
  last_error: string | null;
  notes: string[];
}

export interface DataSourcesHealthResponse {
  sources: DataSourceHealth[];
  checked_live: boolean;
}

export interface CounterfactualResponse {
  feature_name: string;
  delta: number | null;
  original_value: number;
  counterfactual_value: number | null;
}

export interface PredictionResponse {
  id: number;
  ticker: string;
  as_of_date: string;
  direction: number;
  direction_label: string;
  confidence: number;
  regime: number;
  regime_name: string;
  modality_weights: Record<string, number>;
  top_attributions: Record<string, number>;
  counterfactual: CounterfactualResponse | null;
  manipulation_risk: number | null;
  explanation_text: string;
  model_version_id: number | null;
  created_at: string;
}

export interface ModelVersionResponse {
  id: number;
  model_type: string;
  ticker: string;
  metrics: Record<string, number | null>;
  trained_at: string;
  is_active: boolean;
}

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, init);
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const body = await resp.json();
      detail = body.detail || detail;
    } catch {
      // response body wasn't JSON -- fall back to statusText
    }
    throw new ApiError(resp.status, detail);
  }
  return resp.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string; version: string }>("/health"),

  dataSourcesHealth: (live = false) =>
    request<DataSourcesHealthResponse>(`/api/v1/data-sources/health?live=${live}`),

  latestPrediction: (ticker: string) =>
    request<PredictionResponse>(`/api/v1/predictions/${encodeURIComponent(ticker)}/latest`),

  refreshPrediction: (ticker: string) =>
    request<PredictionResponse>(`/api/v1/predictions/${encodeURIComponent(ticker)}/refresh`, {
      method: "POST",
    }),

  predictionHistory: (ticker: string, limit = 20) =>
    request<PredictionResponse[]>(
      `/api/v1/predictions/${encodeURIComponent(ticker)}/history?limit=${limit}`
    ),

  modelVersions: (ticker: string) =>
    request<ModelVersionResponse[]>(`/api/v1/model-versions/${encodeURIComponent(ticker)}`),
};

export { ApiError };
