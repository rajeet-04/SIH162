import { useMemo } from "react";
import type { ThermisEvent } from "@/lib/thermis/types";
import { riskColor } from "./primitives";

interface Props {
  events: ThermisEvent[];
  selectedId?: string | undefined;
  onSelect?: (event: ThermisEvent) => void;
  height?: number;
}

/**
 * Lightweight equirectangular plot of scored events. No map vendor needed —
 * position is derived straight from latitude/longitude.
 */
export function EventMap({ events, selectedId, onSelect, height = 380 }: Props) {
  const points = useMemo(() => {
    if (events.length === 0) return [];
    const lats = events.map((e) => e.latitude);
    const lons = events.map((e) => e.longitude);
    const padLat = 1.2;
    const padLon = 1.2;
    const minLat = Math.min(...lats) - padLat;
    const maxLat = Math.max(...lats) + padLat;
    const minLon = Math.min(...lons) - padLon;
    const maxLon = Math.max(...lons) + padLon;
    const spanLat = Math.max(0.001, maxLat - minLat);
    const spanLon = Math.max(0.001, maxLon - minLon);
    const now = Date.now();
    return events.map((event) => ({
      event,
      x: ((event.longitude - minLon) / spanLon) * 100,
      y: (1 - (event.latitude - minLat) / spanLat) * 100,
      fresh: event.received_at ? now - event.received_at < 6000 : false,
    }));
  }, [events]);

  return (
    <div className="relative w-full overflow-hidden" style={{ height }}>
      <svg
        className="absolute inset-0 size-full text-border"
        aria-hidden="true"
        preserveAspectRatio="none"
        viewBox="0 0 100 100"
      >
        {Array.from({ length: 11 }).map((_, i) => (
          <line
            key={`v${i}`}
            x1={i * 10}
            y1={0}
            x2={i * 10}
            y2={100}
            stroke="currentColor"
            strokeWidth={0.12}
            opacity={0.6}
          />
        ))}
        {Array.from({ length: 11 }).map((_, i) => (
          <line
            key={`h${i}`}
            x1={0}
            y1={i * 10}
            x2={100}
            y2={i * 10}
            stroke="currentColor"
            strokeWidth={0.12}
            opacity={0.6}
          />
        ))}
      </svg>
      <div className="sweep-line pointer-events-none absolute inset-y-0 left-0 w-24 bg-gradient-to-r from-transparent via-primary/8 to-transparent" />
      {points.map(({ event, x, y, fresh }) => {
        const frp = Number(event.evidence?.frp ?? 0);
        const size = Math.max(9, Math.min(26, 9 + Math.sqrt(frp) * 1.7));
        const color = riskColor(event.prediction.risk_level);
        const active = event.event_id === selectedId;
        return (
          <button
            key={event.event_id}
            type="button"
            onClick={() => onSelect?.(event)}
            title={`${event.event_id} · ${event.prediction.final_class}`}
            className={`absolute -translate-x-1/2 -translate-y-1/2 rounded-full transition-transform duration-300 hover:scale-125 focus-visible:outline-2 focus-visible:outline-ring ${
              fresh ? "pulse-ring" : ""
            }`}
            style={{
              left: `${x}%`,
              top: `${y}%`,
              width: size,
              height: size,
              backgroundColor: `color-mix(in oklab, ${color} 72%, transparent)`,
              border: active
                ? `2px solid ${color}`
                : `1px solid color-mix(in oklab, ${color} 60%, white)`,
              transform: active ? "translate(-50%,-50%) scale(1.35)" : "translate(-50%,-50%)",
            }}
          >
            <span className="sr-only">{event.event_id}</span>
          </button>
        );
      })}
      {points.length === 0 && (
        <p className="absolute inset-0 grid place-items-center font-mono text-xs uppercase tracking-[0.2em] text-muted-foreground">
          awaiting scored events
        </p>
      )}
    </div>
  );
}
