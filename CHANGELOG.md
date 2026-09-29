# Changelog

All notable changes to the THERMIS project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] - 2026-09-06

### Added
- **Cross-platform setup and data/model guide** (`README.md`, `docs/data-and-models.md`):
  - Documents fresh-checkout offline operation, tracked model/data artifacts,
    raw-data requirements, rebuild order, credentials, and Windows/macOS/Linux setup.
- **Resumable historical baseline** (`src/thermis/backfill.py`):
  - Imports 90 completed UTC dates for all four FIRMS products with per-product
    coverage, deduplication, retry-safe progress, and optional score refresh.
- **Earthdata access verifier** (`src/thermis/nasa_access.py`):
  - Obtains a short-lived token without persisting a password and verifies a
    bounded protected TIFF header without forwarding bearer credentials to a CDN.
- **Portable PyTorch resolution** (`pyproject.toml`):
  - Uses the CUDA wheel on Windows and ordinary PyTorch wheels on macOS/Linux.
- **`DossierMap` Component** (`app/src/components/thermis/dossier-map.tsx`, `dossier-map-impl.tsx`):
  - Interactive MapLibre GL map centered on the target event coordinates at local tactical zoom (`10.5`).
  - Animated primary target marker featuring risk-colored pulse rings (`.pulse-ring`) and popover detailing class, risk score, and FRP.
  - Flare proximity buffer: dynamic GeoJSON polygon calculation drawing the distance circle to the nearest known flare stack (`nearest_flare_distance_m`).
  - Regional cluster detections: displays neighboring thermal anomalies from the active feed to contrast isolated point sources against spreading fronts.
  - Tactical overlay HUD with coordinate readout (`LAT`, `LON`), `FRP (MW)`, `FLARE PROXIMITY`, `TARGET FOCUS`, and `FIT CLUSTER` controls.
- **`ConsoleMap` Component** (`app/src/components/thermis/console-map.tsx`, `console-map-impl.tsx`):
  - Interactive nationwide MapLibre GL map centered over India (`[79.0, 22.6]`, zoom `4.0`).
  - Renders all active feed detections as interactive GeoJSON circles color-coded by risk level and sized by FRP.
  - Synchronized target selection: clicking any point highlights the anomaly, updates the observation feed, and syncs the evidence panel.
  - Quick action controls: `RECENTRE INDIA` and `FOCUS TARGET`.
  - Tactical bottom status overlay displaying active detection count, target coordinates, and risk tier.
- **Dual-Mode Map Toggles (`OSM Map` / `Radar`)**:
  - Integrated toggle switch in the panel headers of both `/events/$eventId` and `/console` to seamlessly switch between the full OpenStreetMap geographic tile map and the lightweight zero-network SVG radar sweep.
- **Dark Glass Map Popups**:
  - Added `.thermis-map-popup` styling in `app/src/styles.css` matching the platform's dark glass aesthetic.
- **Architecture Documentation**:
  - Created `decisions.md` documenting ADR-001 for geospatial tile map integration.

### Changed
- Upgraded the **Event Dossier** (`/events/$eventId`) location panel to default to the interactive MapLibre map with geographic ground truth.
- Upgraded the **Command Console** (`/console`) event map panel to default to the interactive nationwide MapLibre map with full feed synchronization.
- Consolidated the 6 separate KPI cards on `/console` into a single **Command Telemetry Ribbon** with divider borders and status pips.
- Replaced the landing page icon stat cards with a unified **Session Telemetry Ribbon**.
- Removed banned eyebrow kickers above headings on `/` and `evidence-panel.tsx`, integrating them into inline product badges and subtitle metadata.
- Replaced jittery rolling number animations (`CountUp`) with instantaneous, stable tabular digits.
- Damped the three ambient pastel background blobs in `@utility field` to a single subtle thermal glow, maximizing dark-mode map and table contrast.

### Fixed
- Fixed single-event coordinate collapse in the dossier where single detections rendered as an isolated point on a blank radar sweep with zero geographic reference.
- Fixed `[side-tab]` design anti-pattern in `BackendNotice` (`shell.tsx`) by removing `border-l-4` and introducing unified borders with inline `AlertCircle` indicators.
