import { Link } from "@tanstack/react-router";
import { ContextControls } from "./context-controls";
import type { ThermisEvent } from "@/lib/thermis/types";
import { Meter, RiskChip, classLabel, riskColor } from "./primitives";

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-secondary/60 px-3 py-2">
      <p className="font-mono text-[0.6rem] uppercase tracking-[0.16em] text-muted-foreground">
        {label}
      </p>
      <p className="tabular mt-0.5 text-sm font-semibold text-foreground">{value}</p>
    </div>
  );
}

export function EvidencePanel({ event }: { event: ThermisEvent | null }) {
  if (!event) {
    return (
      <div className="grid h-full min-h-56 place-items-center px-6 py-10 text-center">
        <div>
          <p className="font-display text-base font-semibold text-foreground">Nothing selected</p>
          <p className="mt-1 max-w-xs text-sm text-muted-foreground">
            Pick a point on the map or a row in the feed to see its classification, evidence and
            persistence history.
          </p>
        </div>
      </div>
    );
  }

  const { prediction, evidence } = event;
  const color = riskColor(prediction.risk_level);
  const probabilities = Object.entries(prediction.probabilities ?? {}).sort((a, b) => b[1] - a[1]);

  return (
    <div className="space-y-4 px-4 py-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="font-display text-lg font-semibold capitalize text-foreground">
            {classLabel(prediction.final_class)}
          </h3>
          <div className="mt-1 flex items-center gap-2">
            <span className="font-mono text-xs text-muted-foreground">{event.event_id}</span>
          </div>
        </div>
        <RiskChip level={prediction.risk_level} score={prediction.risk_score} />
      </div>

      <div className="rounded-xl border border-border/60 px-3 py-3">
        <div className="flex items-baseline justify-between">
          <span className="text-xs text-muted-foreground">Confidence</span>
          <span className="tabular font-mono text-2xl font-bold" style={{ color }}>
            {(prediction.confidence * 100).toFixed(1)}%
          </span>
        </div>
        <Meter value={prediction.confidence} color={color} className="mt-2" />
        <div className="mt-2 flex items-baseline justify-between">
          <span className="text-xs text-muted-foreground">Risk score</span>
          <span className="tabular font-mono text-sm font-semibold text-foreground">
            {prediction.risk_score.toFixed(1)}
          </span>
        </div>
        <Meter value={prediction.risk_score / 100} color="var(--risk-high)" className="mt-2" />
      </div>

      <div>
        <p className="font-mono text-[0.62rem] uppercase tracking-[0.16em] text-muted-foreground">
          class probabilities
        </p>
        <ul className="mt-2 space-y-2">
          {probabilities.map(([label, probability]) => (
            <li key={label}>
              <div className="flex items-center justify-between text-[0.74rem]">
                <span className="capitalize text-foreground">{classLabel(label)}</span>
                <span className="tabular text-muted-foreground">
                  {(probability * 100).toFixed(1)}%
                </span>
              </div>
              <Meter value={probability} color={color} className="mt-1" />
            </li>
          ))}
        </ul>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <Stat label="FRP" value={Number(evidence?.frp ?? 0).toFixed(1)} />
        <Stat
          label="flare dist"
          value={
            evidence?.nearest_flare_distance_m == null
              ? "unknown"
              : `${(Number(evidence.nearest_flare_distance_m) / 1000).toFixed(2)} km`
          }
        />
        <Stat label="prior 7d" value={String(evidence?.prior_detections_7d ?? 0)} />
        <Stat label="prior 30d" value={String(evidence?.prior_detections_30d ?? 0)} />
        <Stat label="prior 90d" value={String(evidence?.prior_detections_90d ?? 0)} />
        <Stat label="label src" value={String(evidence?.label_source ?? "—")} />
      </div>

      {event.event_id.startsWith("firms-") && (
        <ContextControls key={event.event_id} event={event} />
      )}
      <div className="flex items-center justify-between gap-2">
        <span
          className={`font-mono text-[0.62rem] uppercase tracking-[0.16em] ${
            prediction.review_required ? "text-primary" : "text-muted-foreground"
          }`}
        >
          {prediction.review_required ? "review required" : "auto-cleared"}
        </span>
        <Link
          to="/events/$eventId"
          params={{ eventId: event.event_id }}
          className="rounded-lg bg-primary px-3 py-1.5 text-xs font-semibold text-primary-foreground transition-colors hover:bg-primary/90"
        >
          Open dossier
        </Link>
      </div>
    </div>
  );
}
