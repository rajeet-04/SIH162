import { lazy, Suspense } from "react";
import { ClientOnly } from "@tanstack/react-router";
import type { ThermisEvent } from "@/lib/thermis/types";

const HeatMapImpl = lazy(() => import("./heat-map-impl"));

function MapSkeleton({ height }: { height: number }) {
  return (
    <div
      className="grid place-items-center bg-secondary/50 font-mono text-[0.66rem] uppercase tracking-[0.18em] text-muted-foreground"
      style={{ height }}
    >
      loading map
    </div>
  );
}

/** Browser-only heat map; maplibre-gl cannot render during SSR. */
export function HeatMap({ events, height = 460 }: { events: ThermisEvent[]; height?: number }) {
  return (
    <ClientOnly fallback={<MapSkeleton height={height} />}>
      <Suspense fallback={<MapSkeleton height={height} />}>
        <HeatMapImpl events={events} height={height} />
      </Suspense>
    </ClientOnly>
  );
}
