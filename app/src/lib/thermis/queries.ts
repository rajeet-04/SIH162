import { queryOptions, useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { thermis } from "./client";
import type { ThermisEvent } from "./types";

export const healthQuery = queryOptions({
  queryKey: ["thermis", "health"],
  queryFn: thermis.health,
  refetchInterval: 5_000,
  retry: 1,
});

export const metricsQuery = queryOptions({
  queryKey: ["thermis", "metrics"],
  queryFn: thermis.metrics,
  refetchInterval: 30_000,
  retry: 1,
});

export const eventsQuery = queryOptions({
  queryKey: ["thermis", "events"],
  queryFn: thermis.events,
  refetchInterval: 5_000,
  retry: 1,
});

export const eventQuery = (id: string) =>
  queryOptions({
    queryKey: ["thermis", "event", id],
    queryFn: () => thermis.event(id),
    retry: 1,
  });

export const timelineQuery = (id: string) =>
  queryOptions({
    queryKey: ["thermis", "timeline", id],
    queryFn: () => thermis.timeline(id),
    retry: 1,
  });

const WINDOW_SIZE = 140;
const INTERVAL = 5000;

export interface LiveFeed {
  events: ThermisEvent[];
  cursor: number;
  total: number;
  scoredThisSession: number;
  lastTickAt: number | null;
  error: Error | null;
  isConnected: boolean;
}

/**
 * Polls durable observations without manufacturing arrivals on repeated polls.
 */
export function useLiveFeed(enabled = true): LiveFeed {
  const [events, setEvents] = useState<ThermisEvent[]>([]);
  const [cursor, setCursor] = useState(0);
  const [total, setTotal] = useState(0);
  const [scored, setScored] = useState(0);
  const [lastTickAt, setLastTickAt] = useState<number | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const cursorRef = useRef(0);
  const busy = useRef(false);
  const seenIds = useRef(new Set<string>());

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;

    const tick = async () => {
      if (busy.current) return;
      busy.current = true;
      try {
        const payload = await thermis.events();
        if (cancelled) return;
        const added = payload.events.filter((event) => !seenIds.current.has(event.event_id));
        const stamped = payload.events.map((event) => ({
          ...event,
          received_at: seenIds.current.has(event.event_id) ? 0 : Date.now(),
        }));
        payload.events.forEach((event) => seenIds.current.add(event.event_id));
        setEvents(stamped.slice(0, WINDOW_SIZE));
        setTotal(payload.events.length);
        setScored((value) => value + added.length);
        setLastTickAt(Date.now());
        setError(null);
        cursorRef.current = Math.min(WINDOW_SIZE, payload.events.length);
        setCursor(cursorRef.current);
      } catch (caught) {
        if (!cancelled) setError(caught as Error);
      } finally {
        busy.current = false;
      }
    };

    void tick();
    const id = setInterval(() => void tick(), INTERVAL);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [enabled]);

  return {
    events,
    cursor,
    total,
    scoredThisSession: scored,
    lastTickAt,
    error,
    isConnected: error === null && lastTickAt !== null,
  };
}

export function useHealth() {
  return useQuery(healthQuery);
}
