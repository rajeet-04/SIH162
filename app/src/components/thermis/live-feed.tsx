import type { ThermisEvent } from "@/lib/thermis/types";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Meter, RiskChip, classLabel, riskColor } from "./primitives";

export function LiveFeed({
  events,
  selectedId,
  onSelect,
  height = 420,
}: {
  events: ThermisEvent[];
  selectedId?: string | undefined;
  onSelect?: (event: ThermisEvent) => void;
  height?: number;
}) {
  const now = Date.now();
  return (
    <ScrollArea style={{ height }}>
      <ul className="divide-y divide-border/60">
        {events.map((event) => {
          const fresh = event.received_at ? now - event.received_at < 6000 : false;
          const color = riskColor(event.prediction.risk_level);
          return (
            <li key={event.event_id} className={fresh ? "slide-in-row" : undefined}>
              <button
                type="button"
                onClick={() => onSelect?.(event)}
                className={`w-full px-4 py-2.5 text-left transition-colors hover:bg-secondary/70 ${
                  event.event_id === selectedId ? "bg-secondary" : ""
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate font-mono text-[0.72rem] text-foreground">
                    {event.event_id.replace(/^(replay|demo)-/, "")}
                  </span>
                  <RiskChip level={event.prediction.risk_level} />
                </div>
                <div className="mt-1 flex items-center justify-between gap-2 text-[0.72rem] text-muted-foreground">
                  <span className="truncate">{classLabel(event.prediction.final_class)}</span>
                  <span className="tabular">
                    FRP {Number(event.evidence?.frp ?? 0).toFixed(1)} ·{" "}
                    {(event.prediction.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <Meter value={event.prediction.confidence} color={color} className="mt-1.5" />
              </button>
            </li>
          );
        })}
        {events.length === 0 && (
          <li className="px-4 py-10 text-center font-mono text-[0.7rem] uppercase tracking-[0.18em] text-muted-foreground">
            feed idle
          </li>
        )}
      </ul>
    </ScrollArea>
  );
}
