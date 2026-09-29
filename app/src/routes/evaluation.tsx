import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { metricsQuery } from "@/lib/thermis/queries";
import { useLiveFeed } from "@/lib/thermis/queries";
import { CountUp, GlassPanel, Meter, PanelHeader } from "@/components/thermis/primitives";
import { RiskMix, SparkStrip } from "@/components/thermis/charts";
import { BackendNotice, SiteHeader, StatusRail } from "@/components/thermis/shell";

export const Route = createFileRoute("/evaluation")({
  head: () => ({
    meta: [
      { title: "Model Evaluation — THERMIS" },
      {
        name: "description",
        content:
          "Stage-by-stage THERMIS model metrics: classification quality, ranking promotion status and the live distribution of scored thermal classes.",
      },
      { property: "og:title", content: "Model Evaluation — THERMIS" },
      {
        property: "og:description",
        content:
          "Inspect THERMIS model performance, stage metrics and live class distribution side by side.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Evaluation,
});

function flatten(value: unknown, prefix = ""): Array<[string, number | string]> {
  if (value === null || value === undefined) return [];
  if (typeof value === "number" || typeof value === "string" || typeof value === "boolean") {
    return [[prefix || "value", typeof value === "boolean" ? String(value) : value]];
  }
  if (Array.isArray(value)) return [[prefix, value.join(", ")]];
  return Object.entries(value as Record<string, unknown>).flatMap(([key, inner]) =>
    flatten(inner, prefix ? `${prefix} · ${key.replace(/_/g, " ")}` : key.replace(/_/g, " ")),
  );
}

function MetricGroup({ title, data }: { title: string; data: unknown }) {
  const rows = flatten(data);
  if (rows.length === 0) return null;
  return (
    <GlassPanel>
      <PanelHeader title={title} meta={<span>{rows.length} metrics</span>} />
      <ul className="divide-y divide-border/60">
        {rows.map(([label, value]) => {
          const numeric = typeof value === "number" ? value : null;
          const ratio = numeric !== null && numeric >= 0 && numeric <= 1 ? numeric : null;
          return (
            <li key={label} className="px-4 py-2.5">
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-[0.78rem] text-muted-foreground">{label}</span>
                <span className="tabular font-mono text-sm font-semibold text-foreground">
                  {numeric !== null ? (
                    <CountUp value={numeric} decimals={numeric % 1 === 0 ? 0 : 4} />
                  ) : (
                    String(value)
                  )}
                </span>
              </div>
              {ratio !== null && <Meter value={ratio} className="mt-1.5" />}
            </li>
          );
        })}
      </ul>
    </GlassPanel>
  );
}

function Evaluation() {
  const metrics = useQuery(metricsQuery);
  const feed = useLiveFeed();
  const data = metrics.data as Record<string, unknown> | undefined;

  const known = new Set(["stage1", "stage2", "ranking", "calibration"]);
  const rest = data
    ? Object.fromEntries(Object.entries(data).filter(([key]) => !known.has(key)))
    : {};

  return (
    <div className="field min-h-screen">
      <SiteHeader />
      <main className="mx-auto max-w-[1600px] space-y-4 px-4 py-5">
        <StatusRail feed={feed} />
        <BackendNotice error={(metrics.error as Error | null) ?? feed.error} />

        <h1 className="font-display text-3xl font-bold tracking-tight text-foreground">
          Model evaluation
        </h1>
        <p className="max-w-2xl text-muted-foreground">
          Every number below is read straight from the running model service — no cached snapshots,
          no hand-written figures.
        </p>

        <section className="grid gap-4 lg:grid-cols-2 xl:grid-cols-3">
          <MetricGroup title="stage 1" data={data?.["stage1"]} />
          <MetricGroup title="stage 2" data={data?.["stage2"]} />
          <MetricGroup title="ranking" data={data?.["ranking"]} />
          <MetricGroup title="calibration" data={data?.["calibration"]} />
          <MetricGroup title="service" data={rest} />
        </section>

        <section className="grid gap-4 xl:grid-cols-2">
          <GlassPanel>
            <PanelHeader title="live class distribution" meta={<span>rolling window</span>} />
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
        </section>
      </main>
    </div>
  );
}
