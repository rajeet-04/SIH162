# SIH26162 MVP

This repository contains the leakage-safe data pipeline and offline-capable MVP
for industrial-fire and persistent-thermal-source classification.

The external source roots are configured in `config/sources.yaml`. Raw source
files are treated as immutable; generated manifests, features, models, and
reports remain under repository output directories.

## Development

Use Python 3.12 with `uv`:

```powershell
uv lock
uv run pytest
uv run ruff check src tests
uv run thermis check-config
```

The chronological newest 10% is reserved as the final ranking set. The older
90% is the only data available for feature fitting, calibration, and threshold
selection.

## MVP workflow

```powershell
# Rebuild only derived artifacts after raw-source changes
.venv\Scripts\thermis.exe prepare-events
.venv\Scripts\thermis.exe build-features
.venv\Scripts\thermis.exe make-splits
.venv\Scripts\thermis.exe train-tabular
.venv\Scripts\thermis.exe train-image --epochs 8 --batch-size 320 --max-records 0 --val-fraction 0.1 --weight-decay 0.0001 --augment
.venv\Scripts\thermis.exe evaluate-ranking

# Run the dashboard in another terminal
cd app
bun run dev

# Rehearse the API, demo bundle, model loading, and CPU image export offline
.venv\Scripts\python.exe scripts\verify_offline.py
```

`train-image` uses CUDA automatically when `torch.cuda.is_available()` is true
and exports a CPU-loadable TorchScript verifier. The current image model is a
bounded smoke model. The running API loads that artifact automatically;
`POST /verify-image` accepts a local image path for image verification, while
`POST /predict` remains the calibrated tabular fusion decision path. See
[reports/model_card.md](reports/model_card.md) for the evaluation status and
limitations.
