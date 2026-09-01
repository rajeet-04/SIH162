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
