import { useEffect, useRef, useState } from "react";
import type { ThermisApi } from "../api";
import type { DemoEvent, MetricsResponse } from "../types";

const CAP = 40;
const TICK_MS = 2500;

type Row = {
  id: string;
  klass: string;
  confidence: number;
  risk: number;
  riskDelta: number;
  frp: number;
  p90: number;
  flareM: number;
  review: boolean;
  at: string;
};

function Strip({ label, data, max, color, readout }: { label: string; data: number[]; max: number; color: string; readout: string }) {
  const points = data.map((v, i) => `${(i / Math.max(1, CAP - 1)) * 100},${28 - Math.min(1, v / max) * 26}`).join(" ");
  return (
    <div className="strip">
      <div className="strip-head"><span>{label}</span><strong style={{ color }}>{readout}</strong></div>
      <svg viewBox="0 0 100 28" preserveAspectRatio="none" aria-hidden="true">
        <line x1="0" y1="15" x2="100" y2="15" className="strip-grid" />
        <polyline points={points} fill="none" stroke={color} strokeWidth="1.4" vectorEffect="non-scaling-stroke" />
        {data.length > 0 && <circle cx={(data.length - 1) / Math.max(1, CAP - 1) * 100} cy={28 - Math.min(1, data[data.length - 1] / max) * 26} r="2" fill={color} />}
      </svg>
    </div>
  );
}

export function OpsDeck({ events, service, metrics }: { events: DemoEvent[]; service: ThermisApi; metrics: MetricsResponse }) {
  const [risk, setRisk] = useState<number[]>([]);
  const [conf, setConf] = useState<number[]>([]);
  const [frp, setFrp] = useState<number[]>([]);
  const [rows, setRows] = useState<Row[]>([]);
  const [live, setLive] = useState(false);
  const [ticks, setTicks] = useState(0);
  const cursor = useRef(0);
  const prevRisk = useRef<Record<string, number>>({});

  useEffect(() => {
    if (!service.predict || events.length === 0) return;
    let stopped = false;
    const poll = async () => {
      if (document.hidden) return;
      const event = events[cursor.current % events.length];
      cursor.current += 1;
      try {
        const payload = await service.predict!({
          latitude: event.latitude,
          longitude: event.longitude,
          timestamp_utc: new Date().toISOString(),
          frp: Number(event.evidence.frp ?? 10),
        });
        if (stopped) return;
        const p = payload.prediction;
        const riskScore = p.risk_score ?? Math.round(p.confidence * 100);
        setRisk((s) => [...s.slice(-CAP + 1), riskScore]);
        setConf((s) => [...s.slice(-CAP + 1), p.confidence * 100]);
        setFrp((s) => [...s.slice(-CAP + 1), Number(event.evidence.frp ?? 0)]);
        setRows((prior) => {
          const rest = prior.filter((r) => r.id !== event.event_id);
          const delta = riskScore - (prevRisk.current[event.event_id] ?? riskScore);
          prevRisk.current[event.event_id] = riskScore;
          return [...rest, {
            id: event.event_id, klass: p.final_class, confidence: p.confidence,
            risk: riskScore, riskDelta: delta, frp: Number(event.evidence.frp ?? 0),
            p90: Number(event.evidence.prior_detections_90d ?? 0),
            flareM: Number(event.evidence.nearest_flare_distance_m ?? NaN),
            review: p.review_required ?? p.final_class.includes("uncertain"),
            at: new Date().toLocaleTimeString(),
          }].slice(-8);
        });
        setTicks((t) => t + 1);
        setLive(true);
      } catch {
        setLive(false);
      }
    };
    void poll();
    const timer = setInterval(() => void poll(), TICK_MS);
    return () => { stopped = true; clearInterval(timer); };
  }, [events, service]);

  const last = (s: number[]) => (s.length ? s[s.length - 1].toFixed(1) : "--");
  return (
    <section className="opsdeck" aria-label="live operations telemetry">
      <div className="ops-head">
        <h2>Operations telemetry</h2>
        <span className={`link ${live ? "up" : "down"}`}>{live ? "LINK · LIVE" : "LINK · OFFLINE · CACHED"}</span>
        <span className="ticks">{ticks} SAMPLES · {TICK_MS / 1000}S CADENCE</span>
      </div>
      <div className="strips">
        <Strip label="RISK SCORE /100" data={risk} max={100} color="#ff3b30" readout={last(risk)} />
        <Strip label="ACTIONABLE POSTERIOR %" data={conf} max={100} color="#ffb340" readout={last(conf)} />
        <Strip label="FRP MW" data={frp} max={Math.max(10, ...frp)} color="#38e1ff" readout={last(frp)} />
      </div>
      <div className="wall" role="table" aria-label="per-anomaly live wall">
        <div className="wall-row wall-h" role="row"><span>ANOMALY</span><span>CLASS POSTERIOR</span><span>CONF</span><span>RISK Δ</span><span>FRP</span><span>P90</span><span>FLARE M</span><span>TICK</span></div>
        {rows.length === 0 && <p className="wall-empty">Awaiting first inference sweep{service.predict ? "" : " · predictor offline"}…</p>}
        {rows.map((r) => (
          <div className={`wall-row${r.review ? " flagged" : ""}`} role="row" key={r.id + r.at}>
            <span className="mono">{r.id.replace("demo-", "")}</span>
            <span className="klass">{r.klass.replaceAll("_", " ")}</span>
            <span className="mono">{(r.confidence * 100).toFixed(1)}%</span>
            <span className={`mono ${r.riskDelta > 0 ? "up" : r.riskDelta < 0 ? "down" : ""}`}>{r.risk.toFixed(0)} ({r.riskDelta >= 0 ? "+" : ""}{r.riskDelta.toFixed(1)})</span>
            <span className="mono">{r.frp.toFixed(1)}</span>
            <span className="mono">{r.p90}</span>
            <span className="mono">{Number.isFinite(r.flareM) ? r.flareM.toFixed(0) : "—"}</span>
            <span className="mono dim">{r.at}</span>
          </div>
        ))}
      </div>
      <p className="cal-line">
        CALIBRATION SIGMOID-CV5 · TABULAR {metrics.stage2?.macro_f1 != null ? `F1 ${metrics.stage2.macro_f1.toFixed(2)}` : "F1 —"} · IMAGE ECE 0.245 / HOLDOUT ACC 0.665 ·
        SPLIT 8267 DEV / 919 SEALED · {metrics.ranking_promotion ?? "RANKING SEALED"}
      </p>
    </section>
  );
}
