import type { DemoEvent } from "../types";
import { TimeMachine } from "./TimeMachine";

const titleize = (value: string) => value.replaceAll("_", " ");

export function EventPanel({ event }: { event: DemoEvent }) {
  const riskScore = event.prediction.risk_score ?? Math.round(event.prediction.confidence * 100);
  const riskLevel = event.prediction.risk_level ?? (riskScore >= 75 ? "high" : "moderate");
  return <div className="evidence-content"><p className="eyebrow">SELECTED EVENT · {event.event_id}</p><h2>{titleize(event.prediction.final_class)}</h2><div className="signal-grid"><div><span>CONFIDENCE</span><strong>{Math.round(event.prediction.confidence * 100)}%</strong></div><div><span>RISK SCORE</span><strong>{riskScore}/100</strong></div><div><span>RISK LEVEL</span><strong className={`risk-${riskLevel}`}>{riskLevel}</strong></div></div>{event.prediction.review_required && <p className="review-banner">REVIEW REQUIRED · evidence paths disagree or context is incomplete</p>}<h3>Evidence trail</h3>{Object.entries(event.evidence).map(([key, value]) => <div className="evidence-row" key={key}><span>{titleize(key)}</span><strong>{typeof value === "number" ? value.toFixed(1) : value}</strong></div>)}<TimeMachine event={event} /><button className="escalate">ESCALATE FOR REVIEW</button></div>;
}
