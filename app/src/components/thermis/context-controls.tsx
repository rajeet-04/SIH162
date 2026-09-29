import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { THERMIS_API_URL } from "@/lib/thermis/client";
import type { ThermisEvent } from "@/lib/thermis/types";

interface SourceItem {
  url: string;
  title?: string;
  name?: string;
}
interface SourceContext {
  source?: string;
  note?: string;
  temperature_c?: number;
  relative_humidity_pct?: number;
  wind_speed_kmh?: number;
  precipitation_mm?: number;
  observed_at_utc?: string;
  scenes?: SourceItem[];
  facilities?: SourceItem[];
}

export function ContextControls({ event }: { event: ThermisEvent }) {
  const [context, setContext] = useState<SourceContext | null>(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");
  const queryClient = useQueryClient();
  const base = `${THERMIS_API_URL}/events/${encodeURIComponent(event.event_id)}`;
  async function load(source: string) {
    setBusy(true);
    setMessage("");
    setContext(null);
    try {
      const response = await fetch(`${base}/context/${source}`);
      if (!response.ok) throw new Error(`${source} evidence unavailable. Try again later.`);
      setContext(await response.json());
    } catch (error) {
      setMessage(String(error));
    } finally {
      setBusy(false);
    }
  }
  async function review(decision: string) {
    setBusy(true);
    try {
      const response = await fetch(`${base}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decision, note }),
      });
      if (!response.ok) throw new Error("Review could not be saved.");
      setMessage(`Review saved: ${decision.replaceAll("_", " ")}`);
      await queryClient.invalidateQueries({ queryKey: ["thermis"] });
    } catch (error) {
      setMessage(String(error));
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="space-y-3 border-t border-border pt-3 text-xs">
      <p>Advisory classification · human review required</p>
      <p>
        Acquired: {event.timestamp_utc ? new Date(event.timestamp_utc).toLocaleString() : "unknown"}
      </p>
      <p>
        {event.evidence?.['history_complete_90d'] === true
          ? 'All four products fetched for the preceding 90 completed UTC dates. Current-day coverage is provisional; satellite coverage does not guarantee detection of every fire.'
          : 'History is partial for this acquisition time; a complete preceding 90-day window is not yet established.'}
      </p>
      <div className="flex flex-wrap gap-2">
        {["weather", "osm", "nasa"].map((source) => (
          <button
            key={source}
            disabled={busy}
            onClick={() => void load(source)}
            className="rounded bg-secondary px-3 py-2 disabled:opacity-50"
          >
            {source === "osm"
              ? "Nearby facilities"
              : source === "nasa"
                ? "Satellite scenes"
                : "Current weather"}
          </button>
        ))}
      </div>
      {context && (
        <div className="space-y-2 rounded bg-secondary p-3">
          <p>{context.source}</p>
          <p>{context.note}</p>
          {context.temperature_c != null && (
            <p>
              {context.temperature_c} °C · humidity {context.relative_humidity_pct}% · wind{" "}
              {context.wind_speed_kmh} km/h · precipitation {context.precipitation_mm} mm
              <br />
              Weather time: {context.observed_at_utc} UTC. Current context, not overpass-time
              conditions.
            </p>
          )}
          {(context.scenes ?? context.facilities ?? []).map((item, i) => (
            <a key={i} href={item.url} target="_blank" rel="noreferrer" className="block underline">
              {item.title ?? item.name}
            </a>
          ))}
          {(context.scenes ?? context.facilities)?.length === 0 && (
            <p>No matching evidence returned.</p>
          )}
        </div>
      )}
      <p>Review: {event.review?.decision.replaceAll("_", " ") ?? "pending"}</p>
      <textarea
        aria-label="Review note"
        value={note}
        maxLength={2000}
        onChange={(e) => setNote(e.target.value)}
        placeholder="Evidence supporting your review"
        className="w-full rounded bg-secondary p-2"
      />
      <div className="flex flex-wrap gap-2">
        {["confirmed", "dismissed", "needs_investigation"].map((decision) => (
          <button
            key={decision}
            disabled={busy}
            onClick={() => void review(decision)}
            className="rounded bg-secondary px-2 py-2 disabled:opacity-50"
          >
            {decision.replaceAll("_", " ")}
          </button>
        ))}
      </div>
      <p role="status">{busy ? "Loading…" : message}</p>
    </div>
  );
}
