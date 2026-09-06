# THERMIS SIH26162 implementation status

Updated: 2026-09-06

This file is the implementation handoff: what is in the branch, what was
verified, and what remains before production deployment. The current branch is
`feature/sih26162-mvp`.

## Completed

### Data and model pipeline

- Source inventory, cleaning, deduplication, feature generation, labels, and
  chronological grouped splits are implemented.
- The newest 10% ranking partition is kept outside fitting, calibration,
  threshold selection, and demo construction. The older 90% is the development
  partition.
- Stage 1 and Stage 2 calibrated CatBoost bundles are shipped under
  `models/tabular/`.
- A bounded ResNet18 image verifier is shipped as CPU-loadable TorchScript under
  `models/image-smoke/`; an experimental U-Net baseline is under
  `models/seg-baseline/` and is not in the decision path.
- The checked-in demo and derived manifests/features/splits make the offline
  MVP runnable without the original raw archives.

### Live monitoring

- NASA FIRMS NOAA-20, NOAA-21, Suomi-NPP, and MODIS ingestion is implemented.
- Five-minute polling, persistent event IDs, SQLite persistence, retries,
  source health, duplicate detection, and advisory scoring are implemented.
- A resumable historical backfill imported all 360 required product-days for
  the 90 completed UTC dates from 2026-06-08 through 2026-09-05.
- Historical rows are marked separately and excluded from the live scoring
  queue. A successful zero-row day is recorded as covered; failed/unavailable
  days are not fabricated as zero detections.
- Existing live events were rescored against the historical baseline while
  preserving review decisions, ingestion timestamps, and cached weather.

### Evidence and dashboard

- Open-Meteo current weather, OpenStreetMap facility context, and NASA CMR/HLS
  scene discovery are available as bounded on-demand evidence.
- Earthdata token setup stores only a short-lived token in ignored `.env`.
  Protected HLS access was verified with a bounded HTTP 206 TIFF-header check.
- MapLibre OSM maps, radar fallback, event selection, nearby context, risk
  display, dossier history, source-health telemetry, and review controls are
  implemented.
- Dashboard polling is five seconds; FIRMS acquisition remains five minutes.

### Reproducibility and safety

- `.env`, `data/secrets/`, `data/live/`, raw archives, and local source roots
  are excluded from Git.
- README, changelog, live runbook, data/model setup guide, design decisions,
  and model card document operation and limitations.
- PyTorch dependency resolution is platform-specific: CUDA wheel on Windows;
  ordinary PyTorch wheels on macOS/Linux. Serving does not require a GPU.
- CI uses `uv sync --frozen`, the Python suite, Ruff, Bun tests, and production
  dashboard build.

## Verification record

The latest local verification completed with:

- Python: 54 tests passed, with one dependency deprecation warning from
  Starlette/httpx.
- Ruff: all checks passed.
- `uv lock --check --offline`: passed before this CI portability correction;
  lock regeneration is part of the final check for this commit.
- Frontend: two Bun tests passed, TypeScript passed, and Vite production build
  passed. Vite reports nonblocking plugin and bundle-size warnings.
- Historical baseline: 360/360 product-days complete.
- NASA: Earthdata authentication succeeded; protected TIFF header returned
  HTTP 206 and passed format verification.

## Remaining work

### Required before production claims

1. Obtain authoritative expert labels for the final ranking set and run a
   locked, temporally held-out evaluation. The current labels are conservative
   rule-derived labels, so the reported tree-model scores are not real-world
   accuracy.
2. Expand the target taxonomy beyond the two shipped classes. Wildfire,
   agricultural burn, industrial source, and uncertain classes require
   aligned labels and validated features.
3. Evaluate the image verifier on representative regions/seasons and reduce
   its measured overconfidence before using it as more than supporting evidence.
4. Build operational authentication, HTTPS, authorization, alert delivery,
   retention policy, SQLite backup/restore, service supervision, and monitoring.
5. Add an India administrative-boundary filter if detections outside India must
   be excluded; the current FIRMS rectangle includes neighboring territory.

### Optional improvements

- Download and verify complete NASA scenes for a selected event instead of only
  checking a bounded TIFF header.
- Add acquisition-time weather and satellite-image feature extraction where
  licensing, latency, and temporal availability permit.
- Add a durable database such as PostgreSQL/PostGIS for multi-user deployment.
- Add browser end-to-end tests and a deployment target with secret management.
- Add scheduled cleanup/retention for historical SQLite observations.

## Fresh-machine runbook

Offline demo:

```bash
uv sync
cd app && bun install --frozen-lockfile && cd ..
uv run python scripts/verify_offline.py
uv run thermis serve --demo --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd app
bun run dev --host 127.0.0.1 --port 5175
```

For live mode, copy `.env.example` to `.env`, configure a FIRMS map key, then
run `uv run thermis serve --live --poll-seconds 300`. For raw-data retraining,
obtain the provider datasets, edit `config/sources.yaml`, and follow the
rebuild order in `docs/data-and-models.md`.
