import { useState } from "react";
import type { DemoEvent } from "../types";

export function TimeMachine({ event }: { event: DemoEvent }) {
  const [windowDays, setWindowDays] = useState<7 | 30 | 90>(90);
  const points = event.timeline_90d.filter((point) => Number(point.days_ago ?? 0) <= windowDays);
  return <section aria-label="thermal history"><div className="section-heading"><h3>Time machine</h3><div className="segmented">{([7, 30, 90] as const).map((days) => <button key={days} className={windowDays === days ? "active" : ""} onClick={() => setWindowDays(days)}>{days}D</button>)}</div></div><div className="timeline" aria-label={`${windowDays}-day thermal history`}>{points.map((point, index) => <div key={`${point.days_ago}-${index}`} title={`${point.days_ago ?? 0} days ago`} style={{ height: `${Math.max(12, Number(point.prior_detections ?? point.frp ?? 1) * 3)}px` }} />)}</div><p className="timeline-note">{windowDays === 90 ? "Long persistence context" : `Recent ${windowDays}-day activity window`} · bars show nearby detections or FRP.</p></section>;
}
