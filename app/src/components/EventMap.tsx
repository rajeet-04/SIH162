import type { DemoEvent } from "../types";

type Props = { events: DemoEvent[]; selectedId?: string; onSelect: (event: DemoEvent) => void };

export function EventMap({ events, selectedId, onSelect }: Props) {
  return (
    <div className="map-surface" aria-label="thermal event map">
      <div className="grid-lines" />
      <div className="map-caption"><strong>GLOBAL THERMAL WATCH</strong><span>WGS84 · OFFLINE TILES</span></div>
      {events.map((event) => <button className={`map-pin ${selectedId === event.event_id ? "selected" : ""}`} style={{ left: `${20 + ((event.longitude + 180) / 360) * 60}%`, top: `${25 + ((90 - event.latitude) / 180) * 50}%` }} key={event.event_id} onClick={() => onSelect(event)} aria-label={event.event_id} title={event.event_id}>{event.prediction.final_class.includes("industrial") ? "!" : "•"}</button>)}
      <p className="map-help">Select a hotspot or event below to open its evidence trail.</p>
    </div>
  );
}
