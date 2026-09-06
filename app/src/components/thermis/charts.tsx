import { useMemo } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ThermisEvent, TimelinePoint } from "@/lib/thermis/types";
import { classLabel, riskColor } from "./primitives";

const tooltipStyle = {
  background: "var(--glass-surface-strong)",
  border: "1px solid var(--border)",
  borderRadius: 12,
  fontSize: 12,
  color: "var(--foreground)",
  boxShadow: "var(--glass-shadow)",
} as const;

export function RiskMix({ events }: { events: ThermisEvent[] }) {
  const data = useMemo(() => {
    const counts = new Map<string, number>();
    for (const event of events) {
      const key = event.prediction.final_class;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return [...counts.entries()]
      .map(([name, count]) => ({ name: classLabel(name), count, raw: name }))
      .sort((a, b) => b.count - a.count);
  }, [events]);

  return (
    <ResponsiveContainer width="100%" height={210}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16, top: 8, bottom: 4 }}>
        <CartesianGrid horizontal={false} stroke="var(--border)" strokeDasharray="2 4" />
        <XAxis type="number" tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} />
        <YAxis
          type="category"
          dataKey="name"
          width={132}
          tick={{ fontSize: 11, fill: "var(--muted-foreground)" }}
        />
        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--muted)" }} />
        <Bar dataKey="count" radius={[0, 6, 6, 0]} fill="var(--chart-1)" isAnimationActive />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function SparkStrip({ events }: { events: ThermisEvent[] }) {
  const data = useMemo(
    () =>
      [...events]
        .slice(0, 60)
        .reverse()
        .map((event, index) => ({
          index,
          frp: Number(event.evidence?.frp ?? 0),
          confidence: event.prediction.confidence * 100,
          risk: event.prediction.risk_score,
        })),
    [events],
  );

  return (
    <ResponsiveContainer width="100%" height={190}>
      <LineChart data={data} margin={{ left: 4, right: 12, top: 10, bottom: 4 }}>
        <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" />
        <XAxis dataKey="index" hide />
        <YAxis tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} width={34} />
        <Tooltip contentStyle={tooltipStyle} />
        <Line
          type="monotone"
          dataKey="confidence"
          stroke="var(--chart-3)"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="risk"
          stroke="var(--chart-1)"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="frp"
          stroke="var(--chart-2)"
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function PersistenceChart({ points }: { points: TimelinePoint[] }) {
  const data = useMemo(
    () =>
      [...points]
        .sort((a, b) => b.days_ago - a.days_ago)
        .map((point) => ({
          label: point.days_ago === 0 ? "acquisition day" : `-${point.days_ago}d`,
          detections: point.prior_detections ?? 0,
          frp: point.frp ?? 0,
        })),
    [points],
  );

  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart data={data} margin={{ left: 4, right: 12, top: 10, bottom: 4 }}>
        <defs>
          <linearGradient id="persistenceFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--chart-1)" stopOpacity={0.45} />
            <stop offset="100%" stopColor="var(--chart-1)" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" />
        <XAxis dataKey="label" tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} />
        <YAxis tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} width={34} />
        <Tooltip contentStyle={tooltipStyle} />
        <Area
          type="monotone"
          dataKey="detections"
          stroke="var(--chart-1)"
          strokeWidth={2}
          fill="url(#persistenceFill)"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function ProbabilityBars({ probabilities }: { probabilities: Record<string, number> }) {
  const data = Object.entries(probabilities).map(([name, value]) => ({
    name: classLabel(name),
    value: value * 100,
  }));
  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ left: 4, right: 12, top: 10, bottom: 4 }}>
        <CartesianGrid vertical={false} stroke="var(--border)" strokeDasharray="2 4" />
        <XAxis dataKey="name" tick={{ fontSize: 10, fill: "var(--muted-foreground)" }} />
        <YAxis tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} width={34} />
        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--muted)" }} />
        <Bar dataKey="value" radius={[6, 6, 0, 0]} fill="var(--chart-1)" />
      </BarChart>
    </ResponsiveContainer>
  );
}

export { riskColor };
