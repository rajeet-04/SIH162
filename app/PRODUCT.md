# PRODUCT.md — THERMIS command dashboard (brief-derived, correct me)

## Product truth
THERMIS triages satellite thermal anomalies for SIH26162 judges and analysts: which
hotspot is an industrial fire, which is persistent flare heat, what risk it carries,
and what evidence supports the call. Offline-capable; live backend when reachable.

## Constraints (non-negotiable)
- Backend: FastAPI on 127.0.0.1:8000 (`/events /predict /metrics /health /timeline`).
  Dashboard polls live; any failure degrades to cached state with a visible LINK flag.
- Offline demo must work with zero network (frozen bundle).
- Vitest strings are load-bearing: event buttons keep `demo-*` ids + confidence,
  EventPanel keeps class heading + evidence keys, TimeMachine keeps 7D/30D/90D +
  `thermal history` region, Evaluation keeps `Promotion readiness` heading.
- No new npm dependencies (hand-rolled SVG telemetry; MapLibre needs tiles = new infra).

## Committed world (2026-09-04 redesign)
Dark ops floor, phosphor-red alert authority, tabular mono telemetry, blinking
hotspot markers (honors `prefers-reduced-motion`), dense but scanable command wall.
