import { lazy, Suspense } from "react";
import { ClientOnly } from "@tanstack/react-router";
import type { ThermisEvent } from "@/lib/thermis/types";

const ConsoleMapImpl = lazy(() => import("./console-map-impl"));

function MapSkeleton({ height }: { height: number }) {
  return (
    <div
      className="grid place-items-center bg-secondary/50 font-mono text-[0.66rem] uppercase tracking-[0.18em] text-muted-foreground"
      style={{ height }}
    >
      <div className="flex items-center gap-2">
        <span className="size-2 animate-ping rounded-full bg-primary" />
        <span>loading geospatial telemetry…</span>
      </div>
    </div>
  );
}

/** Browser-only map for operations console; maplibre-gl cannot render during SSR. */
export function ConsoleMap({
  events,
  selectedId,
  onSelect,
  height = 440,
}: {
  events: ThermisEvent[];
  selectedId?: string | undefined;
  onSelect?: (event: ThermisEvent) => void;
  height?: number;
}) {
  return (
    <ClientOnly fallback={<MapSkeleton height={height} />}>
      <Suspense fallback={<MapSkeleton height={height} />}>
        <ConsoleMapImpl
          events={events}
          selectedId={selectedId}
          onSelect={onSelect}
          height={height}
        />
      </Suspense>
    </ClientOnly>
  );
}
