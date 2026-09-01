# Offline demo bundle

`events.json` is generated from prepared derived Parquet artifacts. It contains
representative event evidence and a 90-day timeline for the offline MVP; raw
source files are not copied into this directory.

Generate it with:

```powershell
uv run python scripts/build_demo_bundle.py
```
