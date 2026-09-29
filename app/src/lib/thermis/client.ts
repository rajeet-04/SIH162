import type {
  EventsResponse,
  HealthResponse,
  ImageVerification,
  MetricsResponse,
  ReplayResponse,
  ThermisEvent,
  TimelineResponse,
} from "./types";

// Always go through the same-origin proxy route; the tunnel/backend does not
// send CORS headers, so direct browser calls are blocked.
export const THERMIS_API_URL = "/api/public/thermis";

export class ThermisApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ThermisApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (!THERMIS_API_URL) {
    throw new ThermisApiError("No THERMIS backend address configured", 0);
  }
  const response = await fetch(`${THERMIS_API_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new ThermisApiError(`Request failed [${response.status}]: ${body}`, response.status);
  }
  return (await response.json()) as T;
}

export const thermis = {
  health: () => request<HealthResponse>("/health"),
  metrics: () => request<MetricsResponse>("/metrics"),
  events: () => request<EventsResponse>("/events"),
  event: (id: string) => request<ThermisEvent>(`/events/${encodeURIComponent(id)}`),
  timeline: (id: string) => request<TimelineResponse>(`/timeline/${encodeURIComponent(id)}`),
  replay: (cursor: number, count = 12, seed = 26162) =>
    request<ReplayResponse>(`/replay?count=${count}&cursor=${cursor}&seed=${seed}`),
  verifyImage: (path: string) =>
    request<{ verification: ImageVerification }>("/verify-image", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),
};
