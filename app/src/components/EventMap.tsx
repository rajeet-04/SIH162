import type { DemoEvent } from "../types";

type Props = { events: DemoEvent[]; selectedId?: string; onSelect: (event: DemoEvent) => void };

const pos = (event: DemoEvent) => ({
  left: `${20 + ((event.longitude + 180) / 360) * 60}%`,
  top: `${25 + ((90 - event.latitude) / 180) * 50}%`,
});

function Marker({ kind }: { kind: string }) {
  if (kind.includes("industrial") && !kind.includes("uncertain"))
    return <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 1 19 10 10 19 1 10Z" fill="currentColor" /></svg>;
  if (kind.includes("persistent"))
    return <svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="8" fill="currentColor" /></svg>;
  return <svg viewBox="0 0 20 20" aria-hidden="true"><rect x="3" y="3" width="14" height="14" fill="currentColor" /></svg>;
}

export function EventMap({ events, selectedId, onSelect }: Props) {
  return (
    <div className="map-surface" aria-label="thermal event map">
      <div className="grid-lines" />
      <div className="map-caption"><strong>GLOBAL THERMAL WATCH</strong><span>WGS84 · OFFLINE TILES</span></div>
      {events.map((event) => {
        const hot = event.prediction.final_class.includes("industrial");
        const flagged = hot || event.prediction.review_required || event.prediction.final_class.includes("uncertain");
        return (
          <button
            className={`map-pin${selectedId === event.event_id ? " selected" : ""}${flagged ? " alert" : ""}`}
            style={pos(event)}
            key={event.event_id}
            onClick={() => onSelect(event)}
            aria-label={event.event_id}
            title={`${event.event_id} · ${event.prediction.final_class}`}
          >
            <Marker kind={event.prediction.final_class} />
          </button>
        );
      })}
      <p className="map-help">Select a hotspot or event below to open its evidence trail.</p>
    </div>
  );
}
