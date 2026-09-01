import type { MetricsResponse } from "../types";

export function EvaluationView({ metrics }: { metrics: MetricsResponse }) {
  return <section className="evaluation" aria-label="model evaluation"><p className="eyebrow">MODEL EVALUATION</p><h2>Promotion readiness</h2><p className="evaluation-status">{metrics.ranking_promotion ?? "offline evaluation loaded"}</p><div className="metric-cards"><div><span>STAGE 1 RECALL</span><strong>{metrics.stage1?.recall?.toFixed(2) ?? "—"}</strong></div><div><span>STAGE 2 F1</span><strong>{metrics.stage2?.macro_f1?.toFixed(2) ?? "—"}</strong></div><div><span>RANKING LABELS</span><strong>0</strong></div></div><p className="muted">The newest 10% ranking set remains sealed. Expert labels are required before recall, calibration, or promotion claims can be made.</p></section>;
}
