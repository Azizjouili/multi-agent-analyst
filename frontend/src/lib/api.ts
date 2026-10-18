// Typed client for the FastAPI endpoints (src/analyst/api.py). Relative URLs:
// in production the API serves this app (same origin); in dev, Vite proxies
// these paths to it (vite.config.ts). Either way, no CORS and no base URL.

export type RunStatus = "running" | "awaiting_review" | "done" | "failed";
export type StageStatus = "running" | "done" | "waiting" | "failed";

export interface Stage {
  node: string;
  status: StageStatus;
  step: number | null;
  started_at: string;
  finished_at: string | null;
}

export interface Critique {
  verdict: "accept" | "revise" | "research";
  scores?: Record<string, number>;
  feedback?: string[] | string;
  missing_topics?: string[];
  budget_exhausted?: boolean;
}

export interface ReviewPayload {
  question?: string;
  draft?: string;
  preview?: string;
  critique?: Critique | null;
  revision_count?: number;
  research_rounds?: number;
}

export interface RunInfo {
  thread_id: string;
  status: RunStatus;
  question: string | null;
  review: ReviewPayload | null;
  report: string | null;
  error: string | null;
  stages?: Stage[]; // additive field; absent from an older API
}

/** An HTTP failure with a readable message. `retryable`: worth trying again. */
export class ApiError extends Error {
  readonly status: number | null;
  readonly retryable: boolean;

  constructor(message: string, status: number | null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    // No response at all (server down, network) or a 5xx (a proxy's 502
    // while the server restarts) is transient; a 4xx won't fix itself.
    this.retryable = status === null || status >= 500;
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError("Can't reach the API server.", null);
  }

  let data: unknown = null;
  try {
    data = await response.json();
  } catch {
    /* an HTML error page from a proxy, or an empty body */
  }
  if (!response.ok) {
    // FastAPI puts the reason in `detail`: a string, or for 422 a list of {msg}.
    let detail = (data as { detail?: unknown } | null)?.detail;
    if (Array.isArray(detail)) detail = detail.map((d: { msg?: string }) => d.msg).join("; ");
    throw new ApiError(typeof detail === "string" ? detail : `HTTP ${response.status}`, response.status);
  }
  return data as T;
}

export const api = {
  startRun: (question: string) => request<RunInfo>("POST", "/runs", { question }),
  getRun: (threadId: string) => request<RunInfo>("GET", `/runs/${encodeURIComponent(threadId)}`),
  review: (threadId: string, approved: boolean, feedback?: string) =>
    request<RunInfo>(
      "POST",
      `/runs/${encodeURIComponent(threadId)}/review`,
      approved ? { approved: true } : { approved: false, feedback },
    ),
  health: () => request<{ status: string; active_runs: number }>("GET", "/health"),
};
