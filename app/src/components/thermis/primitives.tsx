import { useEffect, useRef, useState, type ReactNode } from "react";
import { cn } from "@/lib/utils";

export function GlassPanel({
  className,
  children,
  strong = false,
}: {
  className?: string;
  children: ReactNode;
  strong?: boolean;
}) {
  return (
    <section
      className={cn(strong ? "glass-strong" : "glass", "rounded-2xl overflow-hidden", className)}
    >
      {children}
    </section>
  );
}

export function PanelHeader({
  title,
  meta,
  action,
}: {
  title: string;
  meta?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <header className="flex items-center justify-between gap-3 border-b border-border/60 px-4 py-2.5">
      <h2 className="font-mono text-[0.68rem] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
        {title}
      </h2>
      <div className="flex items-center gap-2 font-mono text-[0.68rem] uppercase tracking-[0.14em] text-muted-foreground">
        {meta}
        {action}
      </div>
    </header>
  );
}

/** Smoothly eases a displayed number toward its target instead of snapping. */
export function CountUp({
  value,
  decimals = 0,
  suffix = "",
  className,
}: {
  value: number;
  decimals?: number;
  suffix?: string;
  className?: string;
}) {
  const [display, setDisplay] = useState(value);
  const from = useRef(value);
  const raf = useRef<number | null>(null);

  useEffect(() => {
    const start = performance.now();
    const origin = from.current;
    const delta = value - origin;
    if (delta === 0) return;
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / 600);
      const eased = 1 - Math.pow(1 - t, 3);
      const next = origin + delta * eased;
      setDisplay(next);
      from.current = next;
      if (t < 1) raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [value]);

  return (
    <span className={cn("tabular", className)}>
      {Number.isFinite(display) ? display.toFixed(decimals) : "—"}
      {suffix}
    </span>
  );
}

export function riskColor(level: string | undefined) {
  const key = (level ?? "").toLowerCase();
  if (key.includes("critical") || key.includes("severe")) return "var(--risk-critical)";
  if (key.includes("high")) return "var(--risk-high)";
  if (key.includes("moderate") || key.includes("medium") || key.includes("elevated"))
    return "var(--risk-moderate)";
  if (key.includes("low")) return "var(--risk-low)";
  return "var(--risk-none)";
}

export function RiskChip({ level, score }: { level: string; score?: number }) {
  const color = riskColor(level);
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 font-mono text-[0.62rem] font-semibold uppercase tracking-[0.12em]"
      style={{ color, backgroundColor: `color-mix(in oklab, ${color} 14%, transparent)` }}
    >
      <span className="size-1.5 rounded-full" style={{ backgroundColor: color }} />
      {level || "unscored"}
      {typeof score === "number" ? ` ${score.toFixed(0)}` : ""}
    </span>
  );
}

export function Meter({
  value,
  color = "var(--primary)",
  className,
}: {
  value: number;
  color?: string;
  className?: string;
}) {
  return (
    <div className={cn("h-1.5 w-full overflow-hidden rounded-full bg-muted", className)}>
      <div
        className="h-full rounded-full transition-[width] duration-700 ease-out"
        style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, backgroundColor: color }}
      />
    </div>
  );
}

export function classLabel(value: string) {
  return value.replace(/_/g, " ");
}
