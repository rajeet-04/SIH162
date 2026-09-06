import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useLiveFeed, metricsQuery } from "@/lib/thermis/queries";
import type { ThermisEvent } from "@/lib/thermis/types";
import {
  CountUp,
  GlassPanel,
  Meter,
  PanelHeader,
  RiskChip,
  classLabel,
  riskColor,
} from "@/components/thermis/primitives";
import { EventMap } from "@/components/thermis/event-map";
import { ConsoleMap } from "@/components/thermis/console-map";
import { LiveFeed } from "@/components/thermis/live-feed";
import { EvidencePanel } from "@/components/thermis/evidence-panel";
import { RiskMix, SparkStrip } from "@/components/thermis/charts";
import { BackendNotice, SiteHeader, StatusRail } from "@/components/thermis/shell";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/console")({
  head: () => ({
    meta: [
      { title: "Command Console — THERMIS" },
      {
        name: "description",
        content:
          "Live THERMIS operations console: scored thermal events on the map, streaming verdicts, evidence breakdown, risk mix and the human review queue.",
      },
      { property: "og:title", content: "Command Console — THERMIS" },
      {
        property: "og:description",
        content:
          "Streaming thermal event verdicts with risk levels, confidence, FRP and persistence evidence.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Console,
});

const HIGH = ["high", "critical", "severe"];

function Console() {
  const feed = useLiveFeed();
  const metrics = useQuery(metricsQuery);
  const [selectedId, setSelectedId] = useState<string | undefined>(undefined);
  const [classFilter, setClassFilter] = useState("all");
  const [mapMode, setMapMode] = useState<"tiles" | "radar">("tiles");

  const filtered = useMemo(
    () =>
      classFilter === "all"
        ? feed.events
        : feed.events.filter((event) =>
            event.prediction.final_class.toLowerCase().includes(classFilter),
          ),
    [feed.events, classFilter],
  );

  const selected: ThermisEvent | null =
    feed.events.find((event) => event.event_id === selectedId) ?? feed.events[0] ?? null;

  const kpis = useMemo(() => {
    const events = feed.events;
    const n = Math.max(1, events.length);
    const review = events.filter(
      (e) =>
        e.prediction.review_required && (!e.review || e.review.decision === "needs_investigation"),
    ).length;
    const high = events.filter((e) =>
      HIGH.some((k) => (e.prediction.risk_level ?? "").toLowerCase().includes(k)),
    ).length;
    const meanConfidence = events.reduce((s, e) => s + e.prediction.confidence, 0) / n;
    const meanFrp = events.reduce((s, e) => s + Number(e.evidence?.frp ?? 0), 0) / n;
    const knownFlare = events.filter((e) => e.evidence?.nearest_flare_distance_m != null);
    const nearFlare = knownFlare.filter(
      (e) => Number(e.evidence?.nearest_flare_distance_m ?? Infinity) < 1500,
    ).length;
    return [
      { label: "in window", value: events.length, decimals: 0 },
      { label: "high risk", value: high, decimals: 0, tone: "var(--risk-high)" },
      { label: "review queue", value: review, decimals: 0, tone: "var(--risk-critical)" },
      { label: "mean confidence", value: meanConfidence * 100, decimals: 1, suffix: "%" },
      { label: "mean FRP", value: meanFrp, decimals: 1 },
      {
        label: "near flare",
        value: knownFlare.length ? (nearFlare / knownFlare.length) * 100 : 0,
        unavailable: knownFlare.length === 0,
        decimals: 0,
        suffix: "%",
      },
    ];
  }, [feed.events]);

  const reviewQueue = useMemo(
    () =>
      feed.events
        .filter(
          (event) =>
            event.prediction.review_required &&
            (!event.review || event.review.decision === "needs_investigation"),
        )
        .sort((a, b) => b.prediction.risk_score - a.prediction.risk_score)
        .slice(0, 40),
    [feed.events],
  );

  return (
    <div className="field min-h-screen">
      <SiteHeader />
      <main className="mx-auto max-w-[1600px] space-y-4 px-4 py-5">
        <StatusRail feed={feed} />
        <BackendNotice error={feed.error} />

        <section className="rounded-2xl border border-border/70 bg-card/80 p-1 shadow-sm backdrop-blur-md">
          <div className="grid grid-cols-2 divide-y divide-border/60 sm:grid-cols-3 sm:divide-y-0 lg:grid-cols-6 lg:divide-x">
            {kpis.map((kpi) => (
              <div key={kpi.label} className="px-4 py-3">
                <div className="flex items-center gap-1.5">
                  {kpi.tone && (
                    <span
                      className="size-1.5 shrink-0 rounded-full"
                      style={{ backgroundColor: kpi.tone }}
                    />
                  )}
                  <span className="text-[0.7rem] text-muted-foreground">{kpi.label}</span>
                </div>
                <p
                  className="tabular mt-1 font-mono text-2xl font-bold tracking-tight"
                  style={{ color: kpi.tone ?? "var(--foreground)" }}
                >
                  {kpi.unavailable
                    ? "unknown"
                    : `${kpi.value.toLocaleString(undefined, {
                        minimumFractionDigits: kpi.decimals,
                        maximumFractionDigits: kpi.decimals,
                      })}${kpi.suffix ?? ""}`}
                </p>
              </div>
            ))}
          </div>
        </section>

        <section className="grid gap-4 xl:grid-cols-[1.6fr_0.9fr_1fr]">
          <GlassPanel>
            <PanelHeader
              title="event map"
              meta={<span>{filtered.length} plotted</span>}
              action={
                <div className="flex items-center gap-2">
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
                  <select
                    value={classFilter}
                    onChange={(event) => setClassFilter(event.target.value)}
                    className="rounded-lg border border-border/70 bg-card px-2 py-1 font-mono text-[0.62rem] uppercase tracking-[0.12em] text-foreground"
                  >
                    <option value="all">all classes</option>
                    <option value="industrial">industrial</option>
                    <option value="persistent">persistent</option>
                    <option value="emergency">emergency</option>
                    <option value="uncertain">uncertain</option>
                  </select>
                </div>
              }
            />
            {mapMode === "tiles" ? (
              <ConsoleMap
                events={filtered}
                selectedId={selected?.event_id}
                onSelect={(event) => setSelectedId(event.event_id)}
                height={380}
              />
            ) : (
              <EventMap
                events={filtered}
                selectedId={selected?.event_id}
                onSelect={(event) => setSelectedId(event.event_id)}
                height={380}
              />
            )}
          </GlassPanel>

          <GlassPanel>
            <PanelHeader title="observation feed" meta={<span>5s refresh · advisory</span>} />
            <LiveFeed
              events={filtered}
              selectedId={selected?.event_id}
              onSelect={(event) => setSelectedId(event.event_id)}
              height={380}
            />
          </GlassPanel>

          <GlassPanel>
            <PanelHeader title="evidence" meta={<span>selected event</span>} />
            <ScrollArea style={{ height: 380 }}>
              <EvidencePanel event={selected} />
            </ScrollArea>
          </GlassPanel>
        </section>

        <section className="grid gap-4 xl:grid-cols-3">
          <GlassPanel>
            <PanelHeader title="class mix" meta={<span>rolling window</span>} />
            <div className="px-2 py-3">
              <RiskMix events={feed.events} />
            </div>
          </GlassPanel>

          <GlassPanel>
            <PanelHeader title="confidence · risk · frp" meta={<span>last 60 verdicts</span>} />
            <div className="px-2 py-3">
              <SparkStrip events={feed.events} />
            </div>
          </GlassPanel>

          <GlassPanel>
            <PanelHeader title="review queue" meta={<span>{reviewQueue.length} flagged</span>} />
            <ScrollArea style={{ height: 214 }}>
              <ul className="divide-y divide-border/60">
                {reviewQueue.map((event) => (
                  <li key={event.event_id}>
                    <button
                      type="button"
                      onClick={() => setSelectedId(event.event_id)}
                      className="w-full px-4 py-2 text-left transition-colors hover:bg-secondary/70"
                    >
                      <div className="flex items-center justify-between gap-2 text-[0.74rem]">
                        <span className="truncate capitalize text-foreground">
                          {classLabel(event.prediction.final_class)}
                        </span>
                        <RiskChip level={event.prediction.risk_level} />
                      </div>
                      <Meter
                        value={event.prediction.risk_score / 100}
                        color={riskColor(event.prediction.risk_level)}
                        className="mt-1.5"
                      />
                    </button>
                  </li>
                ))}
                {reviewQueue.length === 0 && (
                  <li className="px-4 py-10 text-center font-mono text-[0.7rem] uppercase tracking-[0.18em] text-muted-foreground">
                    queue clear
                  </li>
                )}
              </ul>
            </ScrollArea>
          </GlassPanel>
        </section>

        <GlassPanel>
          <PanelHeader
            title="model posture"
            meta={
              <span>{String(metrics.data?.ranking_promotion ?? metrics.data?.status ?? "—")}</span>
            }
          />
          <div className="grid gap-3 px-4 py-3 sm:grid-cols-2 lg:grid-cols-4">
            {Object.entries({
              ...(metrics.data?.stage1 ?? {}),
            })
              .slice(0, 8)
              .map(([key, value]) => (
                <div key={`s1-${key}`} className="rounded-xl bg-secondary/60 px-3 py-2">
                  <p className="font-mono text-[0.58rem] uppercase tracking-[0.14em] text-muted-foreground">
                    stage 1 · {key.replace(/_/g, " ")}
                  </p>
                  <p className="tabular mt-0.5 text-sm font-semibold text-foreground">
                    {typeof value === "number" ? value.toFixed(3) : String(value)}
                  </p>
                </div>
              ))}
            {!metrics.data && (
              <p className="font-mono text-xs uppercase tracking-[0.16em] text-muted-foreground">
                metrics unavailable
              </p>
            )}
          </div>
        </GlassPanel>
      </main>
    </div>
  );
}
