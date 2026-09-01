export type DemoEvent = {
  event_id: string;
  latitude: number;
  longitude: number;
  prediction: { final_class: string; confidence: number; model_version: string };
  evidence: Record<string, number | string>;
  timeline_90d: Array<Record<string, number>>;
};

export type EventsResponse = { events: DemoEvent[] };
