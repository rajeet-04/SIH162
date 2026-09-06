import type { ThermisEvent } from "./types";

export function validLocation(event: Pick<ThermisEvent, "latitude" | "longitude">) {
  return Number.isFinite(event.latitude) && Math.abs(event.latitude) <= 90 &&
    Number.isFinite(event.longitude) && Math.abs(event.longitude) <= 180;
}

export function nearbyObservations(events: ThermisEvent[], target: ThermisEvent, radiusKm = 25) {
  if (!validLocation(target)) return [];
  const rad = Math.PI / 180;
  return events.filter(event => {
    if (!validLocation(event) || event.event_id === target.event_id) return false;
    const dlat = (event.latitude - target.latitude) * rad;
    const dlon = (event.longitude - target.longitude) * rad;
    const a = Math.sin(dlat / 2) ** 2 + Math.cos(target.latitude * rad) *
      Math.cos(event.latitude * rad) * Math.sin(dlon / 2) ** 2;
    return 2 * 6371.0088 * Math.asin(Math.sqrt(Math.min(1, a))) <= radiusKm;
  });
}

export function flareDistanceLabel(distance: unknown) {
  if (typeof distance !== "number" || !Number.isFinite(distance) || distance < 0) return "unknown";
  return distance < 1000 ? `${Math.round(distance)}m` : `${(distance / 1000).toFixed(1)}km`;
}
