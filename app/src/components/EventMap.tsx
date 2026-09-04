import { useEffect, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { DemoEvent } from "../types";

type Props = { events: DemoEvent[]; selectedId?: string; onSelect: (event: DemoEvent) => void };

const OSM_STYLE = {
  version: 8 as const,
  sources: {
    osm: {
      type: "raster" as const,
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      maxzoom: 19,
      attribution: "© OpenStreetMap contributors",
    },
  },
  layers: [{ id: "osm", type: "raster" as const, source: "osm" }],
};

function webglAvailable() {
  try {
    const canvas = document.createElement("canvas");
    return !!canvas.getContext("webgl2", { failIfMajorPerformanceCaveat: true });
  } catch {
    return false;
  }
}

function MarkerGlyph({ kind }: { kind: string }) {
  if (kind.includes("industrial") && !kind.includes("uncertain"))
    return <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 1 19 10 10 19 1 10Z" fill="currentColor" /></svg>;
  if (kind.includes("persistent"))
    return <svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="8" fill="currentColor" /></svg>;
  return <svg viewBox="0 0 20 20" aria-hidden="true"><rect x="3" y="3" width="14" height="14" fill="currentColor" /></svg>;
}

const pos = (event: DemoEvent) => ({
  left: `${20 + ((event.longitude + 180) / 360) * 60}%`,
  top: `${25 + ((90 - event.latitude) / 180) * 50}%`,
});

const flagged = (event: DemoEvent) =>
  event.prediction.final_class.includes("industrial") ||
  event.prediction.review_required ||
  event.prediction.final_class.includes("uncertain");

export function EventMap({ events, selectedId, onSelect }: Props) {
  const container = useRef<HTMLDivElement | null>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const markers = useRef<maplibregl.Marker[]>([]);
  const [failed, setFailed] = useState(!webglAvailable());
  const select = useRef(onSelect);
  select.current = onSelect;

  useEffect(() => {
    if (failed || !container.current || map.current) return;
    let cancelled = false;
    try {
      const instance = new maplibregl.Map({
        container: container.current,
        style: OSM_STYLE,
        center: [78.9, 22.5],
        zoom: 4,
        attributionControl: { compact: true },
      });
      instance.on("error", () => { if (!cancelled) { setFailed(true); instance.remove(); } });
      instance.on("load", () => { if (!cancelled) map.current = instance; else instance.remove(); });
    } catch {
      if (!cancelled) setFailed(true);
    }
    return () => { cancelled = true; };
  }, [failed]);

  useEffect(() => () => { markers.current.forEach((m) => m.remove()); markers.current = []; map.current?.remove(); map.current = null; }, []);

  useEffect(() => {
    const instance = map.current;
    if (failed || !instance) return;
    markers.current.forEach((m) => m.remove());
    markers.current = events.map((event) => {
      const el = document.createElement("button");
      el.className = `map-pin${selectedId === event.event_id ? " selected" : ""}${flagged(event) ? " alert" : ""}`;
      el.setAttribute("aria-label", event.event_id);
      el.title = `${event.event_id} · ${event.prediction.final_class}`;
      el.innerHTML = event.prediction.final_class.includes("persistent")
        ? '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="8" fill="currentColor"/></svg>'
        : event.prediction.final_class.includes("industrial") && !event.prediction.final_class.includes("uncertain")
          ? '<svg viewBox="0 0 20 20"><path d="M10 1 19 10 10 19 1 10Z" fill="currentColor"/></svg>'
          : '<svg viewBox="0 0 20 20"><rect x="3" y="3" width="14" height="14" fill="currentColor"/></svg>';
      el.addEventListener("click", () => select.current(event));
      return new maplibregl.Marker({ element: el }).setLngLat([event.longitude, event.latitude]).addTo(instance);
    });
  }, [events, selectedId, failed]);

  if (failed) {
    return (
      <div className="map-surface" aria-label="thermal event map">
        <div className="grid-lines" />
        <div className="map-caption"><strong>INDIA THERMAL WATCH</strong><span>WGS84 · OFFLINE TILES</span></div>
        {events.map((event) => (
          <button
            className={`map-pin${selectedId === event.event_id ? " selected" : ""}${flagged(event) ? " alert" : ""}`}
            style={pos(event)}
            key={event.event_id}
            onClick={() => onSelect(event)}
            aria-label={event.event_id}
            title={`${event.event_id} · ${event.prediction.final_class}`}
          >
            <MarkerGlyph kind={event.prediction.final_class} />
          </button>
        ))}
        <p className="map-help">Select a hotspot or event below to open its evidence trail.</p>
      </div>
    );
  }

  return (
    <div className="map-surface live" aria-label="thermal event map">
      <div ref={container} className="osm-canvas" />
      <div className="map-caption"><strong>INDIA THERMAL WATCH</strong><span>OPENSTREETMAP · LIVE TILES</span></div>
      <p className="map-help">Select a hotspot or event below to open its evidence trail.</p>
    </div>
  );
}
