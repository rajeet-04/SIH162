import { useEffect, useState } from "react";
import { api, type ThermisApi } from "./api";
import type { DemoEvent } from "./types";
import "./styles.css";

export function App({ service = api }: { service?: ThermisApi }) {
  const [events, setEvents] = useState<DemoEvent[]>([]);
  const [selected, setSelected] = useState<DemoEvent | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    service.listEvents().then((payload) => setEvents(payload.events)).catch(() => setError("API unavailable"));
  }, [service]);

  return (
    <main className="shell">
      <header><div><p className="eyebrow">SIH26162 · OFFLINE COMMAND VIEW</p><h1>THERMIS</h1><p>Industrial thermal intelligence for early action.</p></div><span className="status">● LOCAL MODEL</span></header>
      <section className="stats"><div><span>ACTIVE EVENTS</span><strong>{events.length}</strong></div><div><span>MODEL</span><strong>v0.1</strong></div><div><span>DATA MODE</span><strong>OFFLINE</strong></div></section>
      {error && <p className="error">{error}</p>}
      <section className="workspace">
        <div className="map-panel"><div className="panel-title"><span>EVENT MAP</span><span>90D · ALL REGIONS</span></div><div className="map-surface"><div className="grid-lines" /><p>Select an event to inspect evidence</p>{events.map((event) => <button className="map-pin" style={{ left: `${20 + (event.longitude + 180) / 360 * 60}%`, top: `${25 + (90 - event.latitude) / 180 * 50}%` }} key={event.event_id} onClick={() => setSelected(event)} aria-label={event.event_id}>{event.prediction.final_class.includes("industrial") ? "!" : "•"}</button>)}</div><div className="event-list">{events.map((event) => <button key={event.event_id} onClick={() => setSelected(event)}>{event.event_id.replace("demo-", "")} <span>{Math.round(event.prediction.confidence * 100)}%</span></button>)}</div></div>
        <aside className="evidence">{selected ? <><p className="eyebrow">SELECTED EVENT</p><h2>{selected.prediction.final_class.replaceAll("_", " ")}</h2><div className="confidence"><strong>{Math.round(selected.prediction.confidence * 100)}%</strong><span>confidence</span></div><h3>Evidence trail</h3>{Object.entries(selected.evidence).map(([key, value]) => <div className="evidence-row" key={key}><span>{key.replaceAll("_", " ")}</span><strong>{typeof value === "number" ? value.toFixed(1) : value}</strong></div>)}<h3>90-day timeline</h3><div className="timeline">{selected.timeline_90d.map((point, index) => <div key={index} style={{ height: `${Math.max(12, Number(point.prior_detections ?? point.frp ?? 1) * 3)}px` }} />)}</div><button className="escalate">ESCALATE FOR REVIEW</button></> : <div className="empty"><h2>Investigation panel</h2><p>Choose a mapped event to see its classification, confidence, source evidence, and persistence history.</p></div>}</aside>
      </section>
    </main>
  );
}
