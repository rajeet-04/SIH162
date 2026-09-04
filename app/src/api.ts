import type { DemoEvent, EventsResponse, HealthResponse, MetricsResponse } from "./types";

export type PredictRequest = {
  latitude: number;
  longitude: number;
  timestamp_utc: string;
  frp: number;
  brightness_temperature?: number;
  frp_uncertainty?: number;
};

export type PredictResponse = { prediction: DemoEvent["prediction"] };

export type ThermisApi = {
  listEvents: () => Promise<EventsResponse>;
  getEvent: (id: string) => Promise<DemoEvent>;
  getMetrics?: () => Promise<MetricsResponse>;
  getHealth?: () => Promise<HealthResponse>;
  predict?: (request: PredictRequest) => Promise<PredictResponse>;
  listReplay?: (count: number) => Promise<EventsResponse>;
};

export const api: ThermisApi = {
  listEvents: () => fetch("/events").then((response) => response.json()),
  getEvent: (id) => fetch(`/events/${id}`).then((response) => response.json()),
  getMetrics: () => fetch("/metrics").then((response) => response.json()),
  getHealth: () => fetch("/health").then((response) => response.json()),
  predict: (request) =>
    fetch("/predict", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(request),
    }).then((response) => response.json()),
  listReplay: (count) => fetch(`/replay?count=${count}`).then((response) => response.json()),
};
