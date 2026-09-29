# Architecture & Design Decisions (THERMIS)

This document records the architectural and product design decisions for the THERMIS platform.

---

## ADR-001: Geospatial Tile Map Integration (MapLibre GL + OSM) for Dossier and Console

- **Date**: 2026-09-06
- **Status**: Accepted & Implemented
- **Components**: `app/src/components/thermis/`, `app/src/routes/console.tsx`, `app/src/routes/events.$eventId.tsx`

### Context & Problem
Early project constraints (`PRODUCT.md`) prioritized zero-dependency offline telemetry, which led to the creation of `EventMap`—a hand-rolled equirectangular SVG coordinate scatter plot with an animated sweep line.

However, during operational triage:
1. **Dossier Context Collapse**: In the Event Dossier (`/events/$eventId`), passing a single detection caused the bounding box to collapse around a single coordinate, rendering an isolated dot floating on an empty radar grid with zero geographic, terrain, road, or infrastructure context.
2. **Console Operational Expectations**: On the Command Console (`/console`), the card titled "EVENT MAP" lacked geographic base tiles, making it impossible for operators to discern whether thermal swarms were located in agricultural belts (e.g. Punjab residue burning), industrial corridors (e.g. Gujarat refineries), or forested regions.
3. **Existing Capability**: The project already had `maplibre-gl` installed and configured for the landing page (`HeatMap`), meaning the technical foundation was present but underutilized on primary triage screens.

### Decisions

1. **Dedicated MapLibre Implementations with Dark OSM Raster Tiles**:
   - Created **`DossierMap`** (`dossier-map.tsx`, `dossier-map-impl.tsx`) for event-level tactical investigation.
   - Created **`ConsoleMap`** (`console-map.tsx`, `console-map-impl.tsx`) for nationwide operations and triage.
   - Used OpenStreetMap raster tiles with dark tactical desaturation (`raster-saturation: -0.4`, `raster-brightness-max: 0.94`, `raster-contrast: 0.08`) to preserve the command dark glass aesthetic.

2. **Client-Only Architecture for SSR Safety**:
   - Wrapped implementations with `@tanstack/react-router` `ClientOnly` and React `Suspense` with an animated pulse skeleton. This prevents `window is not defined` or WebGL context failures during server-side rendering in TanStack Start / Nitro.

3. **Dual-Mode Toggle (`OSM Map` / `Radar`)**:
   - Maintained operational flexibility and offline resilience by providing an `OSM Map` vs `Radar` toggle in the panel headers of both the Dossier and Command Console.
   - Operators can use the rich geographic tile map when online, and instantly switch to the zero-network radar plot if tile services are unreachable.

4. **Flare Proximity Visualization**:
   - Implemented dynamic GeoJSON circle computation in `dossier-map-impl.tsx` to render a dashed buffer radius representing distance to the nearest known gas flare stack (`nearest_flare_distance_m`).

5. **Bidirectional Feed Synchronization on `/console`**:
   - Clicking any anomaly marker on the map selects that event, smoothly shifts the animated target marker, highlights the item in the observation feed, and updates the evidence panel.
   - Added **`FOCUS TARGET`** and **`RECENTRE INDIA`** action controls.

### Consequences

- **Positive**:
  - Immediate operational ground truth: analysts can distinguish between flare stacks, industrial facilities, and agricultural/wildfire activity.
  - Retained 100% offline capability via the radar mode toggle.
  - Full synchronization between spatial map, observation feed, and evidence panels.
  - Zero new npm dependencies added (leveraged existing `maplibre-gl` installation).
- **Considerations**:
  - OSM tile fetching requires internet access unless an offline tile server or vector tile package is configured. The `Radar` fallback mode ensures uninterrupted operation without internet.

---

## ADR-002: Impeccable Craft Refactoring & Elimination of AI Slop

- **Date**: 2026-09-06
- **Status**: Accepted & Implemented
- **Components**: `app/src/components/thermis/shell.tsx`, `app/src/routes/index.tsx`, `app/src/routes/console.tsx`, `app/src/components/thermis/evidence-panel.tsx`, `app/src/styles.css`

### Context & Problem
An automated scan using the Impeccable design detector (`detect-antipatterns.mjs`) and manual review against `craft-floor.md` identified several distinct AI code generation hallmarks and design anti-patterns:
1. **Mechanical Violation (`[side-tab]`)**: Thick 4px colored borders (`border-l-4 border-l-primary` / `destructive`) on rounded glass callout cards in `shell.tsx`—the most recognizable signature of LLM-generated UIs.
2. **Hero Metric Card Fatigue**: Repetitive grids of identical floating glass cards with small uppercase labels and oversized numbers on both the Console and Landing page.
3. **Eyebrow / Kicker Antipattern**: Banned uppercase kickers floating above main headings (`Smart India Hackathon 26162` above `h1` in `index.tsx`, and event IDs stacked over titles in `evidence-panel.tsx`).
4. **Jittery Rolling Counters**: `CountUp` number animation running on every click, creating visual noise and cognitive friction during rapid anomaly triage.
5. **Background Fog**: Three overlapping multi-color radial gradient blobs (`@utility field`) washing out dark contrast and making maps and tables look milky.

### Decisions
1. **Eliminated `side-tab` Callouts**: Replaced `border-l-4` with unified borders (`border-destructive/30 bg-destructive/8`) and inline alert icon badges (`AlertCircle`).
2. **Replaced Stat Cards with Mission Telemetry Ribbons**:
   - On `/console`, consolidated the 6 separate KPI cards into a single unified Command Telemetry Ribbon with vertical border dividers and status pips.
   - On `/`, transformed the 4 icon tiles into a cohesive Session Telemetry Ribbon.
3. **Restored Heading Authority**:
   - Replaced the landing page kicker with an integrated pill badge (`SIH 26162 Operational Pilot`).
   - Integrated event IDs as clean subtitle metadata beneath the classification title in `evidence-panel.tsx`.
4. **Instant Tabular Telemetry**: Replaced jittery `CountUp` animations with instantaneous, steady tabular mono digits (`tabular-nums`).
5. **Damped Background Noise**: Reduced three pastel radial blobs to a single subtle thermal field (`radial-gradient(50rem 30rem at 50% -5%...)`) to maximize contrast and map legibility.

### Consequences
- Impeccable design detector passes with **0 anti-patterns found**.
- Increased spatial efficiency and information density matching real space/defense mission consoles.
- Eliminated visual jitter and cognitive delay during operational triage.

