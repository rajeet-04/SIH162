# Retraining verification — 2026-09-09

Run: `data/runs/retrain-20260909-v1/run_report.json`.
Branch: `codex/retraining-upgrade`. Production model promotion remains blocked.

## Coverage recovery and second run

`fetch-osm --reuse-from data/cleaned/osm-20260909-v1 --output
data/cleaned/osm-20260909-v2` now reuses valid successful responses, rejects a
different snapshot date, and requests missing areas into a new output directory.
Surat (285 features) and Raipur (106) both recovered successfully. The six-area
inventory now contains 2,085 features. It is still not nationwide coverage.

Examples at `data/features/retrain-20260909-v2/examples.parquet` were regenerated
from the frozen v1 observations. All 88,400 event identities, timestamps, sites,
episodes and split assignments were verified unchanged. Eligible training rows
increased from 567 to 601 (89 industrial-fire candidates, 512 persistent sources).
Validation now has 201 eligible weak labels, including 15 industrial-fire candidates.

Run `data/runs/retrain-20260909-v2` completed 20 CUDA epochs, batch 256, with no
recorded OOM backoff. CatBoost weak-label macro-F1 is 1.0000 and GRU is 0.8671;
industrial-fire candidate recall is respectively 1.0000 and 0.6000. These are
within-run comparisons on identical examples; v1/v2 aggregate scores use different
eligible validation sets and do not establish a controlled improvement. Promotion
remains blocked. Both v2 models passed CPU reload and normalized-probability checks
on 32 validation observations. Full regression suite: 86 passed, one existing warning.

Historical weather is still absent. The current live client fetches present-time
conditions. The [Open-Meteo historical forecast documentation](https://open-meteo.com/en/docs/historical-forecast-api)
describes a stitched series; run availability must be addressed before claiming
historical serving parity. No weather features were filled by this recovery run.

## Measured results

CUDA training completed 20 epochs with batch size 256 and no recorded OOM backoffs.
Only 567 training examples and 199 validation examples had eligible weak labels.
The prepared dataset contains 88,400 observations; this is not supervised training
on every observation or on all available satellite archives.

| Validation measure | CatBoost | PyTorch GRU |
|---|---:|---:|
| Weak-label macro-F1 | 1.0000 | 0.9038 |
| Industrial-fire candidate recall (13 examples) | 1.0000 | 0.6923 |
| Industrial-fire candidate precision | 1.0000 | 1.0000 |
| Expected calibration error | 0.01283 | 0.01528 |

These scores measure agreement with labeling rules. There are zero independent
reference labels, so accident accuracy is unavailable. The GRU missed four of 13
weak industrial-fire candidates and does not justify promotion. Neither model has
demonstrated coverage of vegetation fires or agricultural burning. Unseen-site and
unseen-region validation subsets lack industrial-fire examples; their aggregate
scores cannot establish industrial-fire generalization.

## Verification completed

- All 84 repository tests passed; one existing Starlette/httpx deprecation warning.
- The 30 retraining tests passed, including sealed-test invariance, past-only
  histories, interrupted training resume, simulated OOM rollback, and CPU reload.
- Both actual saved model artifacts loaded on CPU and produced finite, normalized
  two-class probabilities for 32 validation observations.
- Tests used fresh workspace-local temporary directories, avoiding the previously
  inaccessible shared Windows pytest temporary directory.
- No sealed test evaluation or production-model replacement was performed.

## Artifacts and portability

Keep `catboost.cbm` together with `catboost.metadata.pt`. Keep `gru.pt` for GRU
inference and `checkpoint.pt` for resume. Include `feature_schema.json`,
`dependency_versions.json`, `model_card.md`, and `run_report.json` when transferring
the run. Load only trusted artifacts: the metadata uses PyTorch serialization.
Generated runs are ignored by Git; preserve or transfer them separately.

From the retraining worktree, using a Python environment with project dependencies:

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m pytest -q --basetemp=(Join-Path $PWD (".pytest-tmp-" + [guid]::NewGuid().ToString('N'))) -p no:cacheprovider
```

## Remaining work, in order

1. Six-area OSM recovery is complete in v2. Extend coverage beyond these study
   areas before claiming nationwide industrial-context completeness.
2. Add historical weather with observation-time alignment; all four weather
   features in this run are missing. Establish land-cover coverage and provenance.
3. Align provider-backed fire/flare references with events and independently
   evaluate labels. Unsupported/conflicting examples must remain uncertain.
4. Validate satellite patch band definitions, scaling, georeferencing and labels
   before imagery training. No imagery encoder was trained in this run. MiCASA
   remains deferred as planned.
5. Regenerate versioned examples, retrain both candidates, and compare on identical
   validation rows. Freeze model and thresholds before opening the newest 10%.
6. Complete independent review and live-serving feature parity checks before any
   deployment change. CPU artifact inference alone is not live API integration.

More epochs or higher GPU utilization alone do not resolve the missing reference
labels and source coverage that currently limit this experiment.
