import { useEffect, useMemo, useState } from "react";
import { api, type ThermisApi } from "./api";
import { EventMap } from "./components/EventMap";
import { EventPanel } from "./components/EventPanel";
import { EvaluationView } from "./components/EvaluationView";
import type { DemoEvent, MetricsResponse } from "./types";
import "./styles.css";

export function App({ service = api }: { service?: ThermisApi }) {
  const [events, setEvents] = useState<DemoEvent[]>([]);
  const [selected, setSelected] = useState<DemoEvent | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse>({});
  const [filter, setFilter] = useState("all");
  const [view, setView] = useState<"investigate" | "evaluation">("investigate");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    service.listEvents().then((payload) => setEvents(payload.events)).catch(() => setError("API unavailable"));
    service.getMetrics?.().then(setMetrics).catch(() => undefined);
  }, [service]);

  const filtered = useMemo(() => filter === "all" ? events : events.filter((event) => event.prediction.final_class.includes(filter)), [events, filter]);
  const selectEvent = (event: DemoEvent) => {
    setSelected(event);
    service.getEvent(event.event_id).then(setSelected).catch(() => undefined);
  };
  const reviewCount = events.filter((event) => event.prediction.review_required || event.prediction.final_class.includes("uncertain")).length;

  return <main className="shell"><header><div><p className="eyebrow">SIH26162 · OFFLINE COMMAND VIEW</p><h1>THERMIS</h1><p>Industrial thermal intelligence for early action.</p></div><span className="status">● LOCAL MODEL · READY</span></header><section className="stats"><div><span>VISIBLE EVENTS</span><strong>{filtered.length}</strong></div><div><span>REVIEW QUEUE</span><strong>{reviewCount}</strong></div><div><span>MODEL</span><strong>v0.1</strong></div><div><span>DATA MODE</span><strong>OFFLINE</strong></div></section><nav className="view-tabs" aria-label="dashboard views"><button className={view === "investigate" ? "active" : ""} onClick={() => setView("investigate")}>Investigation</button><button className={view === "evaluation" ? "active" : ""} onClick={() => setView("evaluation")}>Evaluation</button></nav>{error && <p className="error">{error} · using cached state</p>}{view === "evaluation" ? <EvaluationView metrics={metrics} /> : <section className="workspace"><div className="map-panel"><div className="panel-title"><span>EVENT MAP</span><span>90D · ALL REGIONS</span></div><EventMap events={filtered} selectedId={selected?.event_id} onSelect={selectEvent} /><div className="map-controls"><label>CLASS FILTER<select value={filter} onChange={(event) => setFilter(event.target.value)}><option value="all">All signals</option><option value="industrial">Industrial fire</option><option value="persistent">Persistent heat / flare</option><option value="uncertain">Review required</option></select></label></div><div className="event-list">{filtered.map((event) => <button key={event.event_id} onClick={() => selectEvent(event)}>{event.event_id.replace("demo-", "")} <span>{Math.round(event.prediction.confidence * 100)}%</span></button>)}</div></div><aside className="evidence">{selected ? <EventPanel event={selected} /> : <div className="empty"><h2>Investigation panel</h2><p>Choose a mapped event to see its classification, confidence, source evidence, and persistence history.</p></div>}</aside></section>}</main>;
}
