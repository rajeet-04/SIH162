export type RiskLevel = string;

export interface Prediction {
  final_class: string;
  confidence: number;
  risk_score: number;
  risk_level: RiskLevel;
  review_required: boolean;
  probabilities: Record<string, number>;
  model_version: string;
}

export interface Evidence {
  frp?: number;
  prior_detections_7d?: number;
  prior_detections_30d?: number;
  prior_detections_90d?: number;
  nearest_flare_distance_m?: number;
  label_source?: string;
  [key: string]: unknown;
}

export interface TimelinePoint {
  days_ago: number;
  prior_detections?: number;
  frp?: number;
}

export interface ThermisEvent {
  timestamp_utc?: string;
  review?: { decision: string; note: string; reviewed_at_utc: string } | null;
  event_id: string;
  latitude: number;
  longitude: number;
  prediction: Prediction;
  evidence?: Evidence;
  timeline_90d?: TimelinePoint[];
  replay?: boolean;
  /** client-side arrival marker, used for motion only */
  received_at?: number;
}

export interface HealthResponse {
  ingestion?: {
    mode: string;
    enabled: boolean;
    stale?: boolean;
    status?: string;
    last_success_utc?: string;
    next_poll_utc?: string;
    poll_seconds?: number;
    latest_acquisition_utc?: string;
    pending_scores?: number;
    running?: boolean;
  };
  status: string;
  model_version: string;
  offline_demo: boolean;
  image_verifier_loaded: boolean;
  image_verifier_device: string;
}

export interface MetricsResponse {
  stage1?: Record<string, unknown>;
  stage2?: Record<string, unknown>;
  ranking_promotion?: string;
  status?: string;
  [key: string]: unknown;
}

export interface ReplayResponse {
  events: ThermisEvent[];
  total: number;
  cursor: number;
  count: number;
}

export interface EventsResponse {
  events: ThermisEvent[];
}

export interface TimelineResponse {
  scope?: string;
  event_id: string;
  timeline_90d: TimelinePoint[];
}

export interface ImageVerification {
  path: string;
  classes: Record<string, number>;
  predicted_class: string;
  model_version: string;
}
