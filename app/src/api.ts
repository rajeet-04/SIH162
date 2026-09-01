import type { DemoEvent, EventsResponse } from "./types";

export type ThermisApi = {
  listEvents: () => Promise<EventsResponse>;
  getEvent: (id: string) => Promise<DemoEvent>;
};

export const api: ThermisApi = {
  listEvents: () => fetch("/events").then((response) => response.json()),
  getEvent: (id) => fetch(`/events/${id}`).then((response) => response.json()),
};
