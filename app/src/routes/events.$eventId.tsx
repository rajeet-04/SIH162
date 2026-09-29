import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { eventQuery, timelineQuery, useLiveFeed } from "@/lib/thermis/queries";
import { GlassPanel, PanelHeader } from "@/components/thermis/primitives";
import { EvidencePanel } from "@/components/thermis/evidence-panel";
import { EventMap } from "@/components/thermis/event-map";
import { DossierMap } from "@/components/thermis/dossier-map";
import { PersistenceChart, ProbabilityBars } from "@/components/thermis/charts";
import { BackendNotice, SiteHeader, StatusRail } from "@/components/thermis/shell";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/events/$eventId")({
  head: ({ params }) => ({
    meta: [
      { title: `Event ${params.eventId} — THERMIS` },
      {
        name: "description",
        content:
          "Full THERMIS dossier for a single thermal detection: model verdict, class probabilities, fire radiative power, flare proximity and 90-day persistence history.",
      },
      { property: "og:title", content: `Event ${params.eventId} — THERMIS` },
      {
        property: "og:description",
        content: "Model verdict, evidence and persistence history for one thermal detection.",
      },
      { property: "og:type", content: "article" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: EventDossier,
});

function EventDossier() {
  const { eventId } = Route.useParams();
  const feed = useLiveFeed();
  const remote = useQuery(eventQuery(eventId));
  const timeline = useQuery(timelineQuery(eventId));

  const event = remote.data ?? feed.events.find((item) => item.event_id === eventId) ?? null;
  const [mapMode, setMapMode] = useState<"tiles" | "radar">("tiles");

  return (
    <div className="field min-h-screen">
      <SiteHeader />
      <main className="mx-auto max-w-[1400px] space-y-4 px-4 py-5">
        <StatusRail feed={feed} />
        <BackendNotice error={feed.error} />

        <div className="flex items-center justify-between gap-3">
          <h1 className="font-mono text-sm uppercase tracking-[0.18em] text-muted-foreground">
            dossier · {eventId}
          </h1>
          <Link
            to="/console"
            className="glass rounded-lg px-3 py-1.5 text-xs font-semibold text-foreground"
          >
            Back to console
          </Link>
        </div>

        {!event && (
          <GlassPanel className="px-4 py-10 text-center">
            <p className="font-display text-lg font-semibold text-foreground">
              {remote.isLoading ? "Loading event…" : "Event not available"}
            </p>
            <p className="mt-1 text-sm text-muted-foreground">
              {remote.isLoading
                ? "Fetching the stored record from the model service."
                : "This detection is not in the backend store or the live replay window."}
            </p>
          </GlassPanel>
        )}

        {event && (
          <>
            <section className="grid gap-4 lg:grid-cols-[1fr_1.2fr]">
              <GlassPanel>
                <PanelHeader title="verdict & evidence" />
                <EvidencePanel event={event} />
              </GlassPanel>
              <div className="space-y-4">
                <GlassPanel>
                  <PanelHeader
                    title="geospatial location"
                    meta={
                      <span className="font-mono text-[0.62rem] text-muted-foreground">
                        {event.latitude.toFixed(4)}°N, {event.longitude.toFixed(4)}°E
                      </span>
                    }
                    action={
                      <div className="flex items-center rounded-lg border border-border/70 bg-card/60 p-0.5 font-mono text-[0.58rem]">
                        <button
                          type="button"
                          onClick={() => setMapMode("tiles")}
                          className={cn(
                            "rounded px-2 py-0.5 uppercase tracking-wider font-semibold transition-colors",
                            mapMode === "tiles"
                              ? "bg-primary text-primary-foreground"
                              : "text-muted-foreground hover:text-foreground",
                          )}
                        >
                          OSM Map
                        </button>
                        <button
                          type="button"
                          onClick={() => setMapMode("radar")}
                          className={cn(
                            "rounded px-2 py-0.5 uppercase tracking-wider font-semibold transition-colors",
                            mapMode === "radar"
                              ? "bg-primary text-primary-foreground"
                              : "text-muted-foreground hover:text-foreground",
                          )}
                        >
                          Radar
                        </button>
                      </div>
                    }
                  />
                  {mapMode === "tiles" ? (
                    <DossierMap event={event} nearbyEvents={feed.events} height={380} />
                  ) : (
                    <EventMap events={[event]} selectedId={event.event_id} height={380} />
                  )}
                </GlassPanel>
                <GlassPanel>
                  <PanelHeader title="class probabilities" />
                  <div className="px-2 py-3">
                    <ProbabilityBars probabilities={event.prediction.probabilities ?? {}} />
                  </div>
                </GlassPanel>
              </div>
            </section>

            <GlassPanel>
              <PanelHeader
                title="persistence history"
                meta={<span>{timeline.data?.timeline_90d?.length ?? 0} observed days</span>}
              />
              <div className="px-2 py-3">
                <p className="px-2 text-xs text-muted-foreground">{timeline.data?.scope}</p>
                {timeline.data?.timeline_90d?.length ? (
                  <PersistenceChart points={timeline.data.timeline_90d} />
                ) : (
                  <p className="px-2 py-10 text-center font-mono text-[0.7rem] uppercase tracking-[0.18em] text-muted-foreground">
                    no timeline recorded for this detection
                  </p>
                )}
              </div>
            </GlassPanel>
          </>
        )}
      </main>
    </div>
  );
}
