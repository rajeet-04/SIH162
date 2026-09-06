import { createFileRoute, Link } from "@tanstack/react-router";
import { useMemo } from "react";
import { useLiveFeed } from "@/lib/thermis/queries";
import { GlassPanel, RiskChip, classLabel } from "@/components/thermis/primitives";
import { BackendNotice, SiteHeader } from "@/components/thermis/shell";
import { HeatMap } from "@/components/thermis/heat-map";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "THERMIS — Industrial Thermal Intelligence Console" },
      {
        name: "description",
        content:
          "THERMIS scores satellite thermal detections in real time, separating industrial flares from emergency fires and routing uncertain events to human review.",
      },
      { property: "og:title", content: "THERMIS — Industrial Thermal Intelligence Console" },
      {
        property: "og:description",
        content:
          "Live thermal event scoring: risk levels, class probabilities, persistence history and a human review queue.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Landing,
});

function Landing() {
  const feed = useLiveFeed();

  const stats = useMemo(() => {
    const events = feed.events;
    const review = events.filter((e) => e.prediction.review_required).length;
    const high = events.filter((e) =>
      ["high", "critical", "severe"].some((k) =>
        (e.prediction.risk_level ?? "").toLowerCase().includes(k),
      ),
    ).length;
    const meanConfidence =
      events.length > 0
        ? events.reduce((sum, e) => sum + e.prediction.confidence, 0) / events.length
        : 0;
    return { review, high, meanConfidence, total: feed.scoredThisSession };
  }, [feed.events, feed.scoredThisSession]);

  return (
    <div className="field min-h-screen">
      <SiteHeader />
      <main className="mx-auto max-w-[1200px] px-4 pb-20 pt-14">
        <div className="inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
          <span className="size-1.5 rounded-full bg-primary" />
          <span>SIH 26162 Operational Pilot</span>
        </div>
        <h1 className="mt-4 max-w-3xl font-display text-5xl font-bold leading-[1.05] tracking-tight text-foreground">
          Every thermal signal, judged the moment it lands.
        </h1>
        <p className="mt-5 max-w-2xl text-lg text-muted-foreground">
          THERMIS fuses satellite fire radiative power, 90-day persistence history and flare
          proximity into a single verdict — industrial routine, persistent heat, or an emergency
          that needs somebody now.
        </p>

        <div className="mt-8 flex flex-wrap items-center gap-3">
          <Link
            to="/console"
            className="rounded-xl bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground transition-colors hover:bg-primary/90"
          >
            Open the console
          </Link>
          <Link
            to="/evaluation"
            className="glass rounded-xl px-5 py-2.5 text-sm font-semibold text-foreground"
          >
            See model evidence
          </Link>
        </div>

        <div className="mt-10">
          <BackendNotice error={feed.error} />
        </div>

        <section className="mt-8 rounded-2xl border border-border/70 bg-card/80 p-1 backdrop-blur-md shadow-sm">
          <div className="grid grid-cols-2 divide-y divide-border/60 sm:grid-cols-4 sm:divide-x sm:divide-y-0">
            {[
              {
                label: "Scored this session",
                value: stats.total.toLocaleString(),
                meta: "live satellite ingestion",
                tone: undefined,
              },
              {
                label: "High / critical risk",
                value: stats.high.toLocaleString(),
                meta: "immediate action threshold",
                tone: stats.high > 0 ? "text-primary" : undefined,
              },
              {
                label: "Awaiting review",
                value: stats.review.toLocaleString(),
                meta: "human triage queue",
                tone: stats.review > 0 ? "text-accent" : undefined,
              },
              {
                label: "Mean confidence",
                value: `${(stats.meanConfidence * 100).toFixed(1)}%`,
                meta: "calibrated model score",
                tone: undefined,
              },
            ].map((stat) => (
              <div key={stat.label} className="px-5 py-4">
                <span className="text-xs text-muted-foreground">{stat.label}</span>
                <p
                  className={`tabular mt-1 font-mono text-2xl font-bold tracking-tight ${stat.tone ?? "text-foreground"}`}
                >
                  {stat.value}
                </p>
                <p className="mt-0.5 text-[0.7rem] text-muted-foreground/80">{stat.meta}</p>
              </div>
            ))}
          </div>
        </section>

        <GlassPanel className="mt-6">
          <header className="flex items-center justify-between border-b border-border/60 px-4 py-2.5">
            <h2 className="font-mono text-[0.68rem] uppercase tracking-[0.18em] text-muted-foreground">
              thermal intensity · india
            </h2>
            <span className="font-mono text-[0.62rem] uppercase tracking-[0.16em] text-muted-foreground">
              scroll to zoom · drag to pan
            </span>
          </header>
          <HeatMap events={feed.events} height={460} />
        </GlassPanel>

        <GlassPanel className="mt-6">
          <header className="flex items-center justify-between border-b border-border/60 px-4 py-2.5">
            <h2 className="font-mono text-[0.68rem] uppercase tracking-[0.18em] text-muted-foreground">
              live verdict ticker
            </h2>
            <span className="font-mono text-[0.62rem] uppercase tracking-[0.16em] text-muted-foreground">
              refreshing every 5s · advisory classifications
            </span>
          </header>
          <ul className="divide-y divide-border/60">
            {feed.events.slice(0, 6).map((event) => (
              <li
                key={event.event_id}
                className="slide-in-row flex items-center justify-between gap-3 px-4 py-2.5 text-sm"
              >
                <span className="truncate font-mono text-xs text-muted-foreground">
                  {event.event_id.replace(/^replay-/, "")}
                </span>
                <span className="hidden capitalize text-foreground sm:block">
                  {classLabel(event.prediction.final_class)}
                </span>
                <RiskChip level={event.prediction.risk_level} />
              </li>
            ))}
            {feed.events.length === 0 && (
              <li className="px-4 py-8 text-center font-mono text-[0.7rem] uppercase tracking-[0.18em] text-muted-foreground">
                waiting for the scoring feed
              </li>
            )}
          </ul>
        </GlassPanel>
      </main>
    </div>
  );
}
