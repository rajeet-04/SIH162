import { useEffect, useRef } from "react";
import { Map as MapLibreMap, NavigationControl, type GeoJSONSource } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { ThermisEvent } from "@/lib/thermis/types";

const INDIA_CENTER: [number, number] = [79.0, 22.6];
const INDIA_ZOOM = 3.9;

function toGeoJson(events: ThermisEvent[]) {
  return {
    type: "FeatureCollection" as const,
    features: events
      .filter((event) => Number.isFinite(event.latitude) && Number.isFinite(event.longitude))
      .map((event) => ({
        type: "Feature" as const,
        geometry: {
          type: "Point" as const,
          coordinates: [event.longitude, event.latitude],
        },
        properties: {
          weight: Math.max(0.15, Math.min(1, event.prediction.risk_score / 100)),
          frp: Number(event.evidence?.frp ?? 0),
          id: event.event_id,
        },
      })),
  };
}

export default function HeatMapImpl({
  events,
  height = 460,
}: {
  events: ThermisEvent[];
  height?: number;
}) {
  const container = useRef<HTMLDivElement | null>(null);
  const map = useRef<InstanceType<typeof MapLibreMap> | null>(null);
  const ready = useRef(false);
  const latest = useRef(events);
  latest.current = events;

  useEffect(() => {
    if (!container.current || map.current) return;

    const instance = new MapLibreMap({
      container: container.current,
      style: {
        version: 8,
        sources: {
          osm: {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution: "© OpenStreetMap contributors",
          },
        },
        layers: [
          {
            id: "osm",
            type: "raster",
            source: "osm",
            paint: { "raster-saturation": -0.35, "raster-brightness-max": 0.98 },
          },
        ],
      },
      center: INDIA_CENTER,
      zoom: INDIA_ZOOM,
      minZoom: 1,
      maxZoom: 12,
      attributionControl: { compact: true },
    });
    map.current = instance;
    instance.addControl(new NavigationControl({ showCompass: false }), "top-right");

    instance.on("load", () => {
      instance.addSource("thermal", { type: "geojson", data: toGeoJson(latest.current) });

      instance.addLayer({
        id: "thermal-heat",
        type: "heatmap",
        source: "thermal",
        maxzoom: 11,
        paint: {
          "heatmap-weight": ["get", "weight"],
          "heatmap-intensity": ["interpolate", ["linear"], ["zoom"], 1, 0.8, 10, 3],
          "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 1, 14, 6, 34, 11, 60],
          "heatmap-opacity": ["interpolate", ["linear"], ["zoom"], 8, 0.85, 11, 0.35],
          "heatmap-color": [
            "interpolate",
            ["linear"],
            ["heatmap-density"],
            0,
            "rgba(255,255,255,0)",
            0.2,
            "rgba(253, 224, 71, 0.55)",
            0.45,
            "rgba(245, 158, 11, 0.75)",
            0.7,
            "rgba(232, 69, 44, 0.85)",
            1,
            "rgba(159, 18, 57, 0.95)",
          ],
        },
      });

      instance.addLayer({
        id: "thermal-points",
        type: "circle",
        source: "thermal",
        minzoom: 5,
        paint: {
          "circle-radius": [
            "interpolate",
            ["linear"],
            ["zoom"],
            5,
            2.5,
            12,
            ["+", 4, ["*", 0.4, ["sqrt", ["max", ["get", "frp"], 0]]]],
          ],
          "circle-color": [
            "interpolate",
            ["linear"],
            ["get", "weight"],
            0,
            "#F59E0B",
            0.6,
            "#E8452C",
            1,
            "#9F1239",
          ],
          "circle-stroke-color": "#FFFFFF",
          "circle-stroke-width": 1,
          "circle-opacity": ["interpolate", ["linear"], ["zoom"], 5, 0.2, 8, 0.9],
        },
      });

      ready.current = true;
      const source = instance.getSource("thermal") as GeoJSONSource | undefined;
      source?.setData(toGeoJson(latest.current));
    });

    return () => {
      ready.current = false;
      instance.remove();
      map.current = null;
    };
  }, []);

  useEffect(() => {
    if (!map.current || !ready.current) return;
    const source = map.current.getSource("thermal") as GeoJSONSource | undefined;
    source?.setData(toGeoJson(events));
  }, [events]);

  return (
    <div className="relative" style={{ height }}>
      <div ref={container} className="h-full w-full" />
      <button
        type="button"
        onClick={() =>
          map.current?.easeTo({ center: INDIA_CENTER, zoom: INDIA_ZOOM, duration: 700 })
        }
        className="glass-strong absolute left-3 top-3 z-10 rounded-lg px-3 py-1.5 font-mono text-[0.62rem] font-semibold uppercase tracking-[0.14em] text-foreground"
      >
        recentre india
      </button>
      <div className="glass-strong pointer-events-none absolute bottom-3 left-3 z-10 flex items-center gap-2 rounded-lg px-3 py-1.5 font-mono text-[0.58rem] uppercase tracking-[0.14em] text-muted-foreground">
        <span>low</span>
        <span
          className="h-1.5 w-24 rounded-full"
          style={{
            background: "linear-gradient(90deg, rgba(253,224,71,0.7), #F59E0B, #E8452C, #9F1239)",
          }}
        />
        <span>critical</span>
      </div>
    </div>
  );
}
