import { useEffect, useRef } from "react";
import {
  Map as MapLibreMap,
  NavigationControl,
  Marker,
  Popup,
  type GeoJSONSource,
} from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { ThermisEvent } from "@/lib/thermis/types";
import { riskColor } from "./primitives";
import { validLocation } from "@/lib/thermis/map-data";

const INDIA_CENTER: [number, number] = [79.0, 22.6];
const INDIA_ZOOM = 4.0;

interface ConsoleMapImplProps {
  events: ThermisEvent[];
  selectedId?: string | undefined;
  onSelect?: ((event: ThermisEvent) => void) | undefined;
  height?: number;
}

function toGeoJson(events: ThermisEvent[]) {
  return {
    type: "FeatureCollection" as const,
    features: events
      .filter(validLocation)
      .map((e) => ({
        type: "Feature" as const,
        geometry: {
          type: "Point" as const,
          coordinates: [e.longitude, e.latitude],
        },
        properties: {
          id: e.event_id,
          cls: e.prediction.final_class,
          risk: e.prediction.risk_score,
          level: e.prediction.risk_level ?? "moderate",
          frp: Number(e.evidence?.frp ?? 0),
          color: riskColor(e.prediction.risk_level),
        },
      })),
  };
}

export default function ConsoleMapImpl({
  events,
  selectedId,
  onSelect,
  height = 420,
}: ConsoleMapImplProps) {
  const container = useRef<HTMLDivElement | null>(null);
  const map = useRef<InstanceType<typeof MapLibreMap> | null>(null);
  const selectedMarker = useRef<InstanceType<typeof Marker> | null>(null);
  const ready = useRef(false);
  const latestEvents = useRef(events);
  latestEvents.current = events;
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

  const selectedEvent = events.find((e) => e.event_id === selectedId);

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
            paint: {
              "raster-saturation": -0.4,
              "raster-brightness-max": 0.94,
              "raster-contrast": 0.08,
            },
          },
        ],
      },
      center: INDIA_CENTER,
      zoom: INDIA_ZOOM,
      minZoom: 2,
      maxZoom: 16,
      attributionControl: { compact: true },
    });
    map.current = instance;

    instance.addControl(new NavigationControl({ showCompass: false }), "top-right");

    instance.on("load", () => {
      // Add all events as a GeoJSON circle layer
      instance.addSource("console-thermal", {
        type: "geojson",
        data: toGeoJson(latestEvents.current),
      });

      instance.addLayer({
        id: "console-points",
        type: "circle",
        source: "console-thermal",
        paint: {
          "circle-radius": [
            "interpolate",
            ["linear"],
            ["zoom"],
            3,
            3.5,
            6,
            ["+", 4.5, ["*", 0.35, ["sqrt", ["max", ["get", "frp"], 0]]]],
            12,
            ["+", 7, ["*", 0.6, ["sqrt", ["max", ["get", "frp"], 0]]]],
          ],
          "circle-color": ["get", "color"],
          "circle-opacity": 0.82,
          "circle-stroke-color": "#FFFFFF",
          "circle-stroke-width": 1.2,
          "circle-stroke-opacity": 0.9,
        },
      });

      let hoverPopup: InstanceType<typeof Popup> | null = null;

      instance.on("mouseenter", "console-points", (e) => {
        instance.getCanvas().style.cursor = "pointer";
        const feat = e.features?.[0];
        if (!feat || feat.geometry.type !== "Point") return;
        const coords = (feat.geometry as { coordinates: [number, number] }).coordinates;
        const props = feat.properties;

        hoverPopup?.remove();
        hoverPopup = new Popup({
          offset: 12,
          closeButton: false,
          className: "thermis-map-popup",
        })
          .setLngLat(coords)
          .setText(`${props["id"]}\n${props["cls"]}\nRisk: ${props["risk"]}/100 · FRP: ${props["frp"]} MW`)
          .addTo(instance);
      });

      instance.on("mouseleave", "console-points", () => {
        instance.getCanvas().style.cursor = "";
        hoverPopup?.remove();
        hoverPopup = null;
      });

      instance.on("click", "console-points", (e) => {
        const feat = e.features?.[0];
        if (!feat) return;
        const id = feat.properties?.["id"];
        if (id) {
          const clicked = latestEvents.current.find((item) => item.event_id === id);
          if (clicked) {
            onSelectRef.current?.(clicked);
          }
        }
      });

      ready.current = true;

      // Update source data in case events loaded before layer ready
      const source = instance.getSource("console-thermal") as GeoJSONSource | undefined;
      source?.setData(toGeoJson(latestEvents.current));
    });

    return () => {
      ready.current = false;
      selectedMarker.current?.remove();
      selectedMarker.current = null;
      instance.remove();
      map.current = null;
    };
  }, []);

  // Sync data updates
  useEffect(() => {
    if (!map.current || !ready.current) return;
    const source = map.current.getSource("console-thermal") as GeoJSONSource | undefined;
    source?.setData(toGeoJson(events));
  }, [events]);

  // Sync selected event indicator
  useEffect(() => {
    if (!map.current) return;

    if (
      !selectedEvent ||
      !validLocation(selectedEvent)
    ) {
      selectedMarker.current?.remove();
      selectedMarker.current = null;
      return;
    }

    const pinColor = riskColor(selectedEvent.prediction.risk_level);

    if (!selectedMarker.current) {
      const el = document.createElement("div");
      el.className = "relative flex items-center justify-center pointer-events-none";
      el.style.width = "32px";
      el.style.height = "32px";
      el.style.setProperty("--marker-color", pinColor);

      const ping = document.createElement("div");
      ping.className = "absolute inset-0 rounded-full motion-safe:animate-ping opacity-70";
      ping.style.backgroundColor = "var(--marker-color)";

      const ring = document.createElement("div");
      ring.className =
        "relative size-4 rounded-full border-2 border-white shadow-lg flex items-center justify-center";
      ring.style.backgroundColor = "var(--marker-color)";
      ring.style.boxShadow = "0 0 12px var(--marker-color)";

      const centerDot = document.createElement("div");
      centerDot.className = "size-1.5 rounded-full bg-white";

      ring.appendChild(centerDot);
      el.appendChild(ping);
      el.appendChild(ring);

      selectedMarker.current = new Marker({ element: el })
        .setLngLat([selectedEvent.longitude, selectedEvent.latitude])
        .addTo(map.current);
    } else {
      selectedMarker.current.setLngLat([selectedEvent.longitude, selectedEvent.latitude]);
      selectedMarker.current.getElement().style.setProperty("--marker-color", pinColor);
    }
  }, [selectedEvent]);

  const recentreIndia = () => {
    map.current?.easeTo({
      center: INDIA_CENTER,
      zoom: INDIA_ZOOM,
      duration: 700,
    });
  };

  const focusSelected = () => {
    if (
      !map.current ||
      !selectedEvent ||
      !Number.isFinite(selectedEvent.latitude) ||
      !Number.isFinite(selectedEvent.longitude)
    ) {
      return;
    }
    map.current.easeTo({
      center: [selectedEvent.longitude, selectedEvent.latitude],
      zoom: 9.5,
      duration: 700,
    });
  };

  return (
    <div className="relative w-full overflow-hidden" style={{ height }}>
      <div ref={container} className="h-full w-full" />

      {/* Top action controls */}
      <div className="absolute left-3 top-3 z-10 flex items-center gap-2">
        <button
          type="button"
          onClick={recentreIndia}
          className="glass-strong rounded-lg px-2.5 py-1 font-mono text-[0.62rem] font-semibold uppercase tracking-[0.14em] text-foreground transition-colors hover:bg-card"
        >
          recentre india
        </button>
        {selectedEvent && (
          <button
            type="button"
            onClick={focusSelected}
            className="glass-strong rounded-lg px-2.5 py-1 font-mono text-[0.62rem] font-semibold uppercase tracking-[0.14em] text-foreground transition-colors hover:bg-card"
          >
            focus target
          </button>
        )}
      </div>

      {/* Bottom tactical HUD */}
      <div className="glass-strong pointer-events-none absolute bottom-3 left-3 right-3 z-10 flex flex-wrap items-center justify-between gap-2 rounded-xl px-3 py-2 font-mono text-[0.62rem] uppercase tracking-[0.12em] text-muted-foreground">
        <div>
          <span>
            PLOTTED <strong className="text-foreground">{events.length}</strong> ACTIVE DETECTIONS
          </span>
        </div>

        {selectedEvent && (
          <div className="flex items-center gap-3">
            <span>
              FOCUS{" "}
              <strong className="text-foreground">
                {selectedEvent.event_id.replace("firms-", "").slice(0, 8)}
              </strong>
            </span>
            <span>
              LAT <strong className="text-foreground">{selectedEvent.latitude.toFixed(2)}°</strong>
            </span>
            <span>
              LON <strong className="text-foreground">{selectedEvent.longitude.toFixed(2)}°</strong>
            </span>
            <span className="flex items-center gap-1.5">
              <span
                className="size-2 rounded-full"
                style={{ backgroundColor: riskColor(selectedEvent.prediction.risk_level) }}
              />
              <span className="text-foreground font-semibold">
                {selectedEvent.prediction.risk_level ?? "MODERATE"}
              </span>
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
