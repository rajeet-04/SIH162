import { useEffect, useRef } from "react";
import {
  Map as MapLibreMap,
  NavigationControl,
  Marker,
  Popup,
  LngLatBounds,
  type GeoJSONSource,
} from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { ThermisEvent } from "@/lib/thermis/types";
import { riskColor } from "./primitives";
import { nearbyObservations, validLocation, flareDistanceLabel } from "@/lib/thermis/map-data";

interface DossierMapImplProps {
  event: ThermisEvent;
  nearbyEvents?: ThermisEvent[];
  height?: number;
}

function createGeoJsonCircle(center: [number, number], radiusInMeters: number, points = 64) {
  const [lon, lat] = center;
  const km = radiusInMeters / 1000;
  const coordinates: [number, number][] = [];
  const distanceX = km / (111.32 * Math.cos((lat * Math.PI) / 180));
  const distanceY = km / 110.574;

  for (let i = 0; i < points; i++) {
    const theta = (i / points) * (2 * Math.PI);
    const x = distanceX * Math.cos(theta);
    const y = distanceY * Math.sin(theta);
    coordinates.push([lon + x, lat + y]);
  }
  if (coordinates[0]) coordinates.push(coordinates[0]);

  return {
    type: "FeatureCollection" as const,
    features: [
      {
        type: "Feature" as const,
        geometry: {
          type: "Polygon" as const,
          coordinates: [coordinates],
        },
        properties: {
          radius_m: radiusInMeters,
        },
      },
    ],
  };
}

function nearbyToGeoJson(events: ThermisEvent[], targetId: string) {
  return {
    type: "FeatureCollection" as const,
    features: events
      .filter(
        (e) =>
          e.event_id !== targetId && Number.isFinite(e.latitude) && Number.isFinite(e.longitude),
      )
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

export default function DossierMapImpl({
  event,
  nearbyEvents = [],
  height = 380,
}: DossierMapImplProps) {
  const container = useRef<HTMLDivElement | null>(null);
  const map = useRef<InstanceType<typeof MapLibreMap> | null>(null);
  const marker = useRef<InstanceType<typeof Marker> | null>(null);
  const ready = useRef(false);

  const isValidCoord = validLocation(event);
  const rawFlareDist = event.evidence?.nearest_flare_distance_m;
  const flareDist = typeof rawFlareDist === "number" ? rawFlareDist : NaN;
  const hasFlare = Number.isFinite(flareDist) && flareDist >= 0;
  const localEvents = nearbyObservations(nearbyEvents, event);
  const latestNearby = useRef(localEvents);
  latestNearby.current = localEvents;
  // Cap visual circle radius to 25 km so it stays regionally relevant
  const flareRadiusM = hasFlare ? Math.min(flareDist, 25000) : 0;
  const targetCoord: [number, number] = [event.longitude, event.latitude];

  useEffect(() => {
    if (!container.current || map.current || !isValidCoord) return;

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
      center: targetCoord,
      zoom: 10.5,
      minZoom: 2,
      maxZoom: 16,
      attributionControl: { compact: true },
    });
    map.current = instance;

    instance.addControl(new NavigationControl({ showCompass: false }), "top-right");

    instance.on("load", () => {
      // 1. Flare proximity circle layer
      if (hasFlare && flareRadiusM > 0) {
        instance.addSource("flare-circle", {
          type: "geojson", data: createGeoJsonCircle(targetCoord, flareRadiusM),
        });

        instance.addLayer({
          id: "flare-circle-fill",
          type: "fill",
          source: "flare-circle",
          paint: {
            "fill-color": "#f59e0b",
            "fill-opacity": 0.08,
          },
        });

        instance.addLayer({
          id: "flare-circle-line",
          type: "line",
          source: "flare-circle",
          paint: {
            "line-color": "#f59e0b",
            "line-width": 1.5,
            "line-dasharray": [3, 2],
            "line-opacity": 0.75,
          },
        });
      }

      // 2. Surrounding detections layer
      instance.addSource("nearby-detections", {
        type: "geojson",
        data: nearbyToGeoJson(latestNearby.current, event.event_id),
      });

      instance.addLayer({
        id: "nearby-points",
        type: "circle",
        source: "nearby-detections",
        paint: {
          "circle-radius": [
            "interpolate",
            ["linear"],
            ["zoom"],
            4,
            2.5,
            12,
            ["+", 4, ["*", 0.35, ["sqrt", ["max", ["get", "frp"], 0]]]],
          ],
          "circle-color": ["get", "color"],
          "circle-opacity": 0.65,
          "circle-stroke-color": "#FFFFFF",
          "circle-stroke-width": 1,
          "circle-stroke-opacity": 0.8,
        },
      });

      // Hover and click interactions on nearby points
      let hoverPopup: InstanceType<typeof Popup> | null = null;

      instance.on("mouseenter", "nearby-points", (e) => {
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

      instance.on("mouseleave", "nearby-points", () => {
        instance.getCanvas().style.cursor = "";
        hoverPopup?.remove();
        hoverPopup = null;
      });

      instance.on("click", "nearby-points", (e) => {
        const feat = e.features?.[0];
        if (!feat) return;
        const id = feat.properties?.["id"];
        if (id) {
          window.location.href = `/events/${encodeURIComponent(String(id))}`;
        }
      });

      // 3. Animated primary target marker
      const pinColor = riskColor(event.prediction.risk_level);
      const markerEl = document.createElement("div");
      markerEl.className = "relative flex items-center justify-center cursor-pointer group";
      markerEl.style.width = "34px";
      markerEl.style.height = "34px";
      markerEl.setAttribute("aria-label", `Target anomaly: ${event.event_id}`);

      // Outer pulse ring
      const pingRing = document.createElement("div");
      pingRing.className =
        "absolute inset-0 rounded-full motion-safe:animate-ping opacity-60 pointer-events-none";
      pingRing.style.backgroundColor = pinColor;

      // Glow aura
      const glowRing = document.createElement("div");
      glowRing.className = "absolute inset-1 rounded-full opacity-40 blur-[2px]";
      glowRing.style.backgroundColor = pinColor;

      // Solid center badge
      const coreDot = document.createElement("div");
      coreDot.className =
        "relative size-5 rounded-full border-2 border-white shadow-lg flex items-center justify-center";
      coreDot.style.backgroundColor = pinColor;
      coreDot.style.boxShadow = `0 0 14px ${pinColor}`;

      const innerPuck = document.createElement("div");
      innerPuck.className = "size-1.5 rounded-full bg-white";

      coreDot.appendChild(innerPuck);
      markerEl.appendChild(pingRing);
      markerEl.appendChild(glowRing);
      markerEl.appendChild(coreDot);

      const targetPopup = new Popup({
        offset: 20,
        closeButton: false,
        className: "thermis-map-popup",
      }).setText(`TARGET · ${event.event_id}\n${event.prediction.final_class}\nRisk: ${event.prediction.risk_score}/100 (${event.prediction.risk_level})\nFRP: ${Number(event.evidence?.frp ?? 0).toFixed(1)} MW`);

      const targetMarker = new Marker({ element: markerEl })
        .setLngLat(targetCoord)
        .setPopup(targetPopup)
        .addTo(instance);

      marker.current = targetMarker;
      ready.current = true;
    });

    return () => {
      ready.current = false;
      marker.current?.remove();
      marker.current = null;
      instance.remove();
      map.current = null;
    };
  }, [event.event_id]);

  // Update nearby detections if feed updates
  useEffect(() => {
    if (!map.current || !ready.current) return;
    const source = map.current.getSource("nearby-detections") as GeoJSONSource | undefined;
    source?.setData(nearbyToGeoJson(latestNearby.current, event.event_id));
  }, [nearbyEvents, event.event_id]);

  const recentreTarget = () => {
    map.current?.easeTo({
      center: targetCoord,
      zoom: 10.5,
      duration: 650,
    });
  };

  const fitContext = () => {
    if (!map.current) return;
    const bounds = new LngLatBounds(targetCoord, targetCoord);
    localEvents
      .filter((e) => Number.isFinite(e.latitude) && Number.isFinite(e.longitude))
      .slice(0, 40)
      .forEach((e) => bounds.extend([e.longitude, e.latitude]));
    map.current.fitBounds(bounds, { padding: 50, maxZoom: 11, duration: 700 });
  };

  if (!isValidCoord) {
    return (
      <div
        className="grid place-items-center bg-secondary/40 font-mono text-xs text-muted-foreground"
        style={{ height }}
      >
        No geographical coordinates recorded for this event.
      </div>
    );
  }

  return (
    <div className="relative w-full overflow-hidden" style={{ height }}>
      <div ref={container} className="h-full w-full" />

      {/* Top action controls */}
      <div className="absolute left-3 top-3 z-10 flex items-center gap-2">
        <button
          type="button"
          onClick={recentreTarget}
          className="glass-strong rounded-lg px-2.5 py-1 font-mono text-[0.62rem] font-semibold uppercase tracking-[0.14em] text-foreground transition-colors hover:bg-card"
        >
          target focus
        </button>
        {localEvents.length > 0 && (
          <button
            type="button"
            onClick={fitContext}
            className="glass-strong rounded-lg px-2.5 py-1 font-mono text-[0.62rem] font-semibold uppercase tracking-[0.14em] text-muted-foreground transition-colors hover:text-foreground"
          >
            fit nearby (25 km)
          </button>
        )}
      </div>

      {/* Bottom tactical HUD */}
      <div className="glass-strong pointer-events-none absolute bottom-3 left-3 right-3 z-10 flex flex-wrap items-center justify-between gap-2 rounded-xl px-3 py-2 font-mono text-[0.62rem] uppercase tracking-[0.12em] text-muted-foreground">
        <div className="flex items-center gap-3">
          <span>
            LAT <strong className="text-foreground">{event.latitude.toFixed(4)}°N</strong>
          </span>
          <span>
            LON <strong className="text-foreground">{event.longitude.toFixed(4)}°E</strong>
          </span>
          <span>
            FRP{" "}
            <strong className="text-foreground">
              {Number(event.evidence?.frp ?? 0).toFixed(1)} MW
            </strong>
          </span>
        </div>

        <div className="flex items-center gap-3">
          {hasFlare ? (
            <span className={flareDist < 1500 ? "text-accent font-semibold" : ""}>
              FLARE{" "}
              {flareDistanceLabel(flareDist)}{flareDist > 25000 ? " (ring capped at 25 km)" : ""}
            </span>
          ) : (
            <span>FLARE unknown</span>
          )}
          <span className="flex items-center gap-1.5">
            <span
              className="size-2 rounded-full"
              style={{ backgroundColor: riskColor(event.prediction.risk_level) }}
            />
            <span className="text-foreground font-semibold">
              {event.prediction.risk_level ?? "MODERATE"}
            </span>
          </span>
        </div>
      </div>
    </div>
  );
}
