export type DemoEvent = {
  event_id: string;
  latitude: number;
  longitude: number;
  prediction: {
    final_class: string;
    confidence: number;
    model_version: string;
    risk_score?: number;
    risk_level?: string;
    review_required?: boolean;
  };
  evidence: Record<string, number | string>;
  timeline_90d: Array<Record<string, number>>;
};

export type EventsResponse = { events: DemoEvent[] };
export type MetricsResponse = {
  stage1?: Record<string, number>;
  stage2?: Record<string, number>;
  ranking_promotion?: string;
};
export type HealthResponse = {
  status?: string;
  model_version?: string;
  image_verifier_loaded?: boolean;
  image_verifier_device?: string;
};
