import type { DemoEvent, EventsResponse, MetricsResponse } from "./types";

export type ThermisApi = {
  listEvents: () => Promise<EventsResponse>;
  getEvent: (id: string) => Promise<DemoEvent>;
  getMetrics?: () => Promise<MetricsResponse>;
};

export const api: ThermisApi = {
  listEvents: () => fetch("/events").then((response) => response.json()),
  getEvent: (id) => fetch(`/events/${id}`).then((response) => response.json()),
  getMetrics: () => fetch("/metrics").then((response) => response.json()),
};
