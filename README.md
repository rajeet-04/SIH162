# THERMIS — SIH26162 MVP

THERMIS is an offline-capable decision-support MVP for detecting and triaging
industrial fires and persistent thermal sources. It combines a calibrated
tabular model with a supporting image verifier, live NASA FIRMS ingestion,
Open-Meteo weather context, OpenStreetMap facility context, NASA Earthdata
scene discovery, and a MapLibre dashboard.

The shipped classifier is advisory and currently has two trained classes:
`industrial_fire_candidate` and `persistent_industrial_heat_or_flare`. It is
not an autonomous emergency, regulatory, or production decision system. See
the [model card](reports/model_card.md) for measured limitations.

## Quick offline demo

Requirements on Windows, macOS, or Linux:

- Git
- Python 3.12 (managed automatically by `uv`)
- `uv` — <https://docs.astral.sh/uv/>
- Bun — <https://bun.sh/>

```bash
git clone https://github.com/rajeet-04/SIH162.git
cd SIH162
uv sync
cd app && bun install --frozen-lockfile && cd ..
uv run python scripts/verify_offline.py
uv run thermis serve --demo --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd SIH162/app
bun run dev --host 127.0.0.1 --port 5175
```

Open <http://127.0.0.1:5175/console>. The offline demo uses only the tracked
derived demo bundle and checked-in model artifacts; it needs no API keys, GPU,
external datasets, or network after dependencies are installed.

## What is included in Git

The repository includes the reproducible code, configuration templates,
offline demo, prepared manifests/features/labels/splits, and model artifacts:

- `models/tabular/stage1.joblib` and `stage2.joblib` — calibrated CatBoost
  models used by the API.
- `models/image-smoke/image_verifier.ts` — CPU-loadable TorchScript supporting
  image verifier.
- `models/seg-baseline/seg_unet.ts` — experimental segmentation baseline,
  not wired into the decision path.
- `data/demo/events.json` — small offline event bundle.
- `data/manifests/`, `data/cleaned/`, `data/features/`, `data/labels/`, and
  `data/splits/` — derived training artifacts used to reproduce the models.

Large raw archives, live SQLite state, credentials, and local source roots are
intentionally not committed. The checked-in model and demo files are enough to
run the offline MVP on a new computer.

## Live mode

Copy `.env.example` to `.env`, then configure a FIRMS map key. Either set:

```text
FIRMS_MAP_KEY=your_key
```

or put the key alone in the ignored file named by `FIRMS_KEYS_FILE`. Never put
credentials in Git or in frontend code. Run:

```bash
uv run thermis serve --live --poll-seconds 300 --host 127.0.0.1 --port 8000
```

The monitor fetches four FIRMS products every five minutes, deduplicates
observations, scores new rows, records source health, and retries failed
scores. The dashboard polls saved results every five seconds. A live run also
supports on-demand Open-Meteo, OSM, and NASA HLS evidence. NASA protected
imagery requires `NASA_EARTHDATA_TOKEN`; use the interactive helper:

```bash
uv run python -m thermis.nasa_access
```

It prompts for the Earthdata password without saving it, obtains a token, and
stores only the token in ignored `.env`. It verifies a bounded protected TIFF
header request and does not download a full scene.

## Historical baseline

The resumable backfill covers the 90 completed UTC dates before today for all
four FIRMS products. It stores coverage separately from observations, treats a
successful zero-row day as valid, never treats a failed/unavailable day as
zero, deduplicates by persistent event ID, and keeps historical rows out of
the live scoring queue.

```bash
uv run python -m thermis.backfill
uv run python -m thermis.backfill --refresh
```

`--refresh` recomputes existing live scores with the available historical
context while preserving reviews, ingestion timestamps, and cached weather.
The imported live database is local runtime state under `data/live/` and is
not committed.

## Reproducing training

The model split is chronological: the newest 10% is an untouched ranking set;
the older 90% is the only data available for fitting, calibration, and
threshold selection. Training requires the raw source roots described in
`config/sources.yaml`; edit those paths for the machine being used.

```bash
uv run thermis inventory --config config/sources.yaml
uv run thermis prepare-manifests --config config/sources.yaml
uv run thermis prepare-events
uv run thermis build-features
uv run thermis make-splits
uv run thermis train-tabular
uv run thermis train-image --epochs 8 --batch-size 320 --val-fraction 0.1 --weight-decay 0.0001 --augment
uv run thermis evaluate-ranking
```

CUDA is optional. On Windows with a supported NVIDIA driver, the locked CUDA
wheel is used and image training uses CUDA automatically. On macOS and
CPU-only systems, `uv` resolves the non-CUDA PyTorch wheels and training falls
back to CPU. Reduce `--batch-size` for smaller GPUs; the image trainer already
uses batches and exports a CPU-loadable artifact.

## Raw data required for retraining

Raw data is not redistributed by this repository. Obtain it from the original
providers, keep it outside Git, and update `config/sources.yaml`:

- FIRMS CSV exports for live or historical thermal detections.
- FireSat/Fire Atlas image sources used by the image manifest.
- FlareSat facility metadata for industrial/flare proximity context.
- MODIS MCD64A1 burned-area data and Global Fire Atlas archives for future
  source labels; these are not silently treated as fire-class labels.
- MiCASA 3-hourly/daily flux archives are optional environmental context and
  were not consumed as fire-classifier training labels.

For the current verified inventory, source locations and limitations are
recorded in the [model card](reports/model_card.md) and
[data/model setup guide](docs/data-and-models.md). Run `uv run thermis
check-config` after editing paths.

## Verification and documentation

```bash
uv run pytest
uv run ruff check src tests scripts
cd app && bun test && bun run build
```

Operational behavior is documented in [docs/live-monitoring.md](docs/live-monitoring.md).
Design decisions are in [docs/decisions.md](docs/decisions.md); release history
is in [CHANGELOG.md](CHANGELOG.md).
