import { Link } from "@tanstack/react-router";
import type { ReactNode } from "react";
import { Flame, AlertCircle } from "lucide-react";
import { useHealth, type LiveFeed } from "@/lib/thermis/queries";
import { THERMIS_API_URL } from "@/lib/thermis/client";

const NAV = [
  { to: "/", label: "Overview" },
  { to: "/console", label: "Console" },
  { to: "/evaluation", label: "Evaluation" },
] as const;

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-30 border-b border-border/60 bg-background/70 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1600px] items-center justify-between gap-4 px-4 py-3">
        <Link to="/" className="flex items-center gap-2">
          <span className="grid size-8 place-items-center rounded-xl bg-primary text-primary-foreground">
            <Flame className="size-4" />
          </span>
          <span className="font-display text-base font-bold tracking-tight">THERMIS</span>
          <span className="hidden font-mono text-[0.62rem] uppercase tracking-[0.2em] text-muted-foreground sm:inline">
            SIH26162
          </span>
        </Link>
        <nav className="flex items-center gap-1">
          {NAV.map((item) => (
            <Link
              key={item.to}
              to={item.to}
              activeOptions={{ exact: item.to === "/" }}
              activeProps={{ className: "bg-secondary text-foreground" }}
              className="rounded-lg px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}

export function StatusRail({ feed }: { feed: LiveFeed }) {
  const health = useHealth();
  const connected = feed.isConnected && !health.isError;
  const items: Array<[string, ReactNode]> = [
    [
      "link",
      <span
        key="link"
        className="inline-flex items-center gap-1.5"
        style={{ color: connected ? "var(--risk-low)" : "var(--risk-critical)" }}
      >
        <span className="size-1.5 rounded-full" style={{ backgroundColor: "currentColor" }} />
        {connected ? "online" : "unreachable"}
      </span>,
    ],
    ["model", health.data?.model_version ?? "—"],
    ["source", health.data?.ingestion?.mode ?? "unknown"],
    [
      "ingestion",
      health.data?.ingestion?.running
        ? "processing"
        : health.data?.ingestion?.stale
          ? "stale"
          : (health.data?.ingestion?.status ?? "idle"),
    ],
    [
      "FIRMS fetched",
      health.data?.ingestion?.last_success_utc
        ? new Date(health.data.ingestion.last_success_utc).toLocaleString()
        : "never",
    ],
    [
      "newest overpass",
      health.data?.ingestion?.latest_acquisition_utc
        ? new Date(health.data.ingestion.latest_acquisition_utc).toLocaleString()
        : "unknown",
    ],
    ["verifier", health.data?.image_verifier_loaded ? "loaded" : "off"],
    ["visible", `${feed.cursor}/${feed.total || "0"}`],
    ["unique seen", feed.scoredThisSession],
    [
      "last tick",
      feed.lastTickAt ? `${Math.round((Date.now() - feed.lastTickAt) / 1000)}s ago` : "—",
    ],
  ];

  return (
    <div className="glass flex flex-wrap items-center gap-x-6 gap-y-2 rounded-2xl px-4 py-2.5">
      {items.map(([label, value]) => (
        <div key={label} className="flex items-baseline gap-2">
          <span className="font-mono text-[0.6rem] uppercase tracking-[0.18em] text-muted-foreground">
            {label}
          </span>
          <span className="tabular font-mono text-xs font-semibold text-foreground">{value}</span>
        </div>
      ))}
    </div>
  );
}

export function BackendNotice({ error }: { error: Error | null }) {
  if (!THERMIS_API_URL) {
    return (
      <div className="flex items-start gap-3 rounded-xl border border-primary/25 bg-primary/8 px-4 py-3 text-sm">
        <AlertCircle className="mt-0.5 size-4 shrink-0 text-primary" />
        <div>
          <p className="font-semibold text-foreground">No backend address configured</p>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Set <code className="font-mono text-xs">VITE_THERMIS_API_URL</code> to the running
            THERMIS API address to stream live model verdicts into this console.
          </p>
        </div>
      </div>
    );
  }
  if (!error) return null;
  return (
    <div className="flex items-start gap-3 rounded-xl border border-destructive/30 bg-destructive/8 px-4 py-3 text-sm">
      <AlertCircle className="mt-0.5 size-4 shrink-0 text-destructive" />
      <div>
        <p className="font-semibold text-foreground">Backend unreachable</p>
        <p className="mt-0.5 break-words text-xs text-muted-foreground">{error.message}</p>
      </div>
    </div>
  );
}
