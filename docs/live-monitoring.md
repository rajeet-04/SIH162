# Live monitoring pilot

Run from the SIH162-mvp worktree:

```powershell
uv run thermis serve --live --poll-seconds 300
```

In another terminal, from `app`:

```powershell
bun run dev --host 127.0.0.1 --port 5175
```

Open http://localhost:5175/console. The API runs on port 8000. Keep this pilot
bound to loopback: internet deployment requires authentication for reviews and
the existing local image-verification endpoint. Run one API process/worker.

## Implemented phases

1. FIRMS: NOAA20, NOAA21, SNPP and MODIS; two-day overlap, five-minute polling,
   acquisition-time validation, persistent IDs, SQLite observations and scores.
2. Inference: existing frozen model bundles; earlier-only local history within
   5 km; unavailable features remain missing; all live results require review.
3. Enrichment: bounded automatic current-weather requests (12 per cycle), cached
   on-demand current weather, OSM facility context and NASA HLS scene discovery.
4. Product: five-second stored-event refresh, source health, acquisition times,
   evidence controls, saved review decisions and correct 0–100 risk displays.
5. Verification: restart deduplication, invalid/future observations, temporal
   history, source outages, scoring retries and API review tests.

## Credentials

The service loads a local `.env` without overriding process environment values.
Copy `.env.example` if needed. FIRMS defaults to the first key in
`D:/data/firms_india/.map_keys`; it does not rotate keys to evade quotas.
No key is sent to the browser. Open-Meteo and OSM need no key for this pilot.
Set `NASA_EARTHDATA_TOKEN` locally for authenticated CMR requests. An exposed
password is not stored here. Public scene discovery can work without a token;
that does not verify protected download authorization.

## Data and interpretation

The area is a rectangular India window (66E–100E, 4N–39N), including neighboring
territory. Counts are satellite observations, not unique fires. Different
satellites retain separate observations; repeated downloads of the same
product/time/location do not inflate counts. The console displays the latest
140 of up to 500 saved scored observations, not all-time totals.

Persistence covers only observations ingested by this service, not a complete
90-day baseline. No FIRMS detections does not prove absence of fire. Current
Open-Meteo weather is explicitly separate from acquisition-time conditions.
OSM coverage is incomplete; scene discovery is not image-model verification.
The current model has two trained classes and lacks authoritative evaluation
labels. This release neither retrains nor promotes it. Human review does not
automatically become a training label. Distance context remains missing in the
frozen input contract; OSM evidence is displayed separately pending evaluation.

Database: `data/live/events.sqlite3` (ignored by Git). Back it up with SQLite's
backup API while running, or copy it with its WAL after stopping the service.
Raw external datasets and model artifacts are not modified. Failed scores stay
pending for retry. Source failures retain the last successful observations and
surface degraded/stale status. Stop with Ctrl+C. `--demo` remains available for
offline fixtures; `/replay` remains explicit and is not the default feed.

References: https://firms.modaps.eosdis.nasa.gov/api/area/,
https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html,
https://urs.earthdata.nasa.gov/documentation/for_users/user_token

## Verification record — 2026-09-06

- Python: 45 tests passed; full Ruff check passed.
- Frontend: TypeScript check and production build passed; nonblocking bundle-size warning.
- `uv lock --check --offline`: passed (102 packages).
- Initial cycle: NOAA20 988, NOAA21 707, SNPP 842, MODIS 70 observations;
  2,607 stored and scored; zero rejected rows and zero scoring failures.
- Restart cycle completed at 13:21:37 UTC: 2,607 duplicate observations,
  zero new rows, zero pending scores; read-only database counts remained 2,607.
- Browser verified live mode, event selection, acquisition timestamp, current
  weather values and correct risk-score scale.
- Selected event: OSM request succeeded with no nearby facility matches;
  NASA CMR returned one HLSL30 scene. No Earthdata token was configured, so
  protected-data authorization remains unverified.
- Data Analytics quality checks drove separate observation/ingestion timestamps,
  missing-distance display, earlier-only history and duplicate/retry tests.

## Dashboard-change review — 2026-09-06

- Preserved the new MapLibre maps and telemetry ribbon. Fixed TypeScript errors,
  GeoJSON source construction, stale selected-marker color and popup HTML injection.
- Unknown flare distances stay unknown; nearby-map fitting is limited to 25 km
  within the loaded feed, not the entire national observation window.
- Live dossiers now show observed daily counts within 5 km, up to acquisition
  time. Missing days are not invented as zero detections; local history is incomplete.
- Verification: 46 Python tests, two map-data tests, Ruff, TypeScript and production
  build passed. Build retains nonblocking bundle-size warnings.
- Restarted API and dashboard. Browser verified 140 plotted observations, rendered
  OSM tiles, selected dossier, unknown-distance labels and two observed history days.
- Latest checked ingestion cycle: two new rows scored, 2,615 duplicates skipped,
  zero scoring failures and zero pending scores. FIRMS polling remains five minutes;
  the dashboard refreshes saved results every five seconds.
