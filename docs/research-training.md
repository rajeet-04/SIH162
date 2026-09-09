# Research training: source data, labels and thermal novelty

This experimental branch now supports authenticated FIRMS history acquisition and
an unsupervised CUDA denoising autoencoder. The new model ranks unusual thermal
observations. A verified industrial-fire classifier remains unfinished.

## Why the perfect CatBoost score is not trustworthy evidence of classification

The current weak target uses flare/facility distance, preceding active days and
FRP intensity ratios. Those same columns enter CatBoost. Thus 1.0 validation F1
demonstrates reproduction of the label generator, not confirmed incident accuracy.
This is circular evaluation; ordinary statistical overfitting is not established
by the perfect score alone. Removing CatBoost or changing neural architectures
does not create independent truth.

The anomaly model therefore uses no target column, FIRMS type flag, generated
class, facility-label rule, or previous classifier output. It learns a compressed
representation of physical measurements, observation time and location. It can
support triage, but cannot replace classification or establish accident risk.

## Newly acquired data

- VIIRS SNPP standard-processed observations, India-region bounding box
  68E/4N/98E/38N (not an administrative India boundary).
- 2024: 967,250 distinct observations; two duplicate records excluded.
- 2025: 1,020,493 distinct observations; one duplicate record excluded.
- Combined: 1,987,743 observations. Raw five-day CSV chunks and SHA-256 checksums
  are retained next to each annual `observations.parquet` and `manifest.json`.
- Seven of ten configured keys passed NASA metadata preflight; slots 3–5 were
  rejected. The key file was not modified or copied into the repository.
- Requests rotate healthy keys with one-second minimum aggregate spacing.
  HTTP 429 stops all key use. Completed chunks are checksum-verified on resume.
  Credentials and credential-bearing request URLs are excluded from logs.
- All requested date chunks returned valid CSV responses, but 21 calendar dates
  across the two years contain no detections. This does not establish absence of
  fire: satellite, retrieval and cloud coverage remain relevant limitations.

FIRMS documents five-day Area API queries and transaction limits in its
[API reference](https://firms.modaps.eosdis.nasa.gov/api/area/).

## Model and evaluation scope

Eight numeric features: log(1+FRP), brightness temperature, cyclic UTC hour and
month, latitude and longitude. Each has a missing-value indicator. Training-only
robust scaling precedes a 16→32→4→32→16 denoising autoencoder.

The chronological split keeps equal acquisition times together. The two-year run
has 1,589,809 fit rows, 199,034 validation rows, and 198,900 rows excluded as test.
Training covers all of 2024 and extends to 6 April 2025. Validation begins at
2025-04-06 20:27 UTC; this run's test starts at 2025-05-18 08:12 UTC.

The earlier one-year exploratory run evaluated some dates that fall within the
two-year test interval. Consequently that interval is held out from this fit but
is **not a pristine cross-experiment benchmark**. It must not support a final
generalization claim. The original 2026 retraining test cohort has not been
evaluated by this work. No accuracy, fire recall or false-alert rate is claimed.

The checkpoint with lowest validation reconstruction error is saved, alongside
a validation-99th-percentile review threshold. That threshold identifies unusually
large reconstruction errors; it is not a calibrated fire probability or verified
false-positive rate. No spatial holdout performance is established. Single-sensor
calibration and seasonal/site distribution changes may affect scores.

## Imagery and NASA access

The existing Earthdata token returned HTTP 206 and a valid 4,096-byte TIFF header
for a historical HLS tile covering the Surat area. This verifies protected access,
not full imagery download or image-label alignment.

The [FlareSat provider](https://zenodo.org/records/17619196) fire archive was
downloaded and matched published MD5 `3872c9e98561eda7c8b362cc18b14f65`, but lacks
a usable ZIP central directory and ends inside an image entry. Recovery handles
streaming DEFLATE descriptors and individually checks CRC, length and SHA-256.
It recovered 241 complete members into `data/cleaned/fire-patches-crc-v2`.
The provider download and previous incomplete attempts remain preserved.

CRC recovery proves integrity of those entries only. The inspected TIFF has no
georeferencing or band descriptions, and generic raster readers do not expose it
as a normal ten-band geospatial raster. Its tensor layout, band order, scaling,
provider masks and scene/site grouping must be verified before training. Missing
flare imagery also prevents a balanced flare-versus-fire experiment today.

## Run commands

Run from this worktree with the project Python environment active. The commands
read credentials locally; they do not contain credentials.

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m thermis.research_firms --start 2024-01-01 --end 2024-12-31 --output data/cleaned/firms-snpp-2024-v1
python -m thermis.research_firms --start 2025-01-01 --end 2025-12-31 --output data/cleaned/firms-snpp-2025-v1
python -m thermis.anomaly_training --observations data/cleaned/firms-snpp-2024-v1/observations.parquet data/cleaned/firms-snpp-2025-v1/observations.parquet --output data/runs/anomaly-snpp-2024-2025-v1 --device cuda --epochs 20 --batch-size 256 --resume
```

For a new run use a new output directory and omit `--resume`. Checkpoints preserve
model, optimizer, mixed-precision scaler, RNG, active batch size and data/config
identity. Resume rejects changes to data or configuration. GPU memory failures
roll back the interrupted training epoch and reduce the batch size. CPU inference
loads `anomaly.pt` through `thermis.anomaly_training.load_anomaly` and calls
`score(frame)` on normalized FIRMS observations.

Transfer `anomaly.pt`, `run_report.json`, and `model_card.md` together. Keep
`checkpoint.pt` for resume and `validation_rankings.parquet` for research review.
Weights contain preprocessing and feature order. Load only trusted PyTorch files.
These artifacts are ignored by Git and must be transferred separately.

## Saved-run verification (2026-09-09)

The completed two-year artifact loaded independently on CPU and CUDA. Scores on
128 historical observations agreed within `atol=1e-6, rtol=1e-4`. All code cells
in `notebooks/research-data-quality.ipynb` executed successfully. These are runtime
checks, not independent classification validation or live-serving integration.
The full regression suite passed: 98 tests, with one existing Starlette/httpx
deprecation warning, using a fresh workspace-local temporary directory.

On Windows, use an explicitly resolved fresh test directory to avoid permissions
on the shared pytest directory:

```powershell
$researchTestTemp = Join-Path $PWD ('.pytest-tmp-' + [guid]::NewGuid().ToString('N'))
python -m pytest -q --basetemp $researchTestTemp -p no:cacheprovider
```

## Remaining classifier work

Independent event/site reference labels, imagery/label alignment, land-cover and
historical-weather joins, spatial holdout evaluation and serving parity remain
required. More raw FIRMS detections cannot identify confirmed industrial accidents
without that evidence. No current challenger is promoted to the live dashboard.
