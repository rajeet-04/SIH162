# THERMIS SIH26162 model card

## Intended use

THERMIS is an offline-capable decision-support MVP for detecting and triaging
industrial fires and persistent thermal sources. It ranks an incoming thermal
event for analyst review; it is not an autonomous emergency or regulatory
decision system.

## Data and labels

- Canonical event table: 9,186 accepted rows from the prepared structured event
  sources; 6 malformed rows were rejected.
- Supporting sources are tracked in `config/sources.yaml` and the generated
  manifests. Gas-flaring facility points are used for proximity features.
- Labels are conservative, rule-derived candidates using persistence and
  industrial/flare proximity. 7,930 rows remain review-required and 1,256 are
  eligible for supervised training.
- The image verifier uses 47,992 readable Fire/No_Fire images from the local
  Training and Test folders. It is a bounded ResNet18 smoke model, not yet a
  production satellite-image classifier.

## Training and leakage controls

The newest 10% of whole event time groups is reserved as the untouched ranking
set: 8,267 development rows and 919 ranking rows. Ranking rows are excluded
from fitting, calibration, threshold selection, and demo construction. Inside
development, the tabular estimator fits on the oldest 80% of eligible rows and
uses the later 20% for sigmoid calibration diagnostics. Scene-group isolation
is enforced for image records.

## Current artifacts

Stage 1 and Stage 2 are calibrated CatBoost classifiers. Each currently has
739 fit rows, 185 calibration rows, and `ranking_rows_used=0`. The image smoke
verifier was trained for one epoch on 128 balanced-shuffled examples using
CUDA Torch on the available RTX 5050 Laptop GPU and exported as CPU-loadable
TorchScript for the offline demo.

## Evaluation and limitations

The ranking set has no authoritative labels in the available sources, so the
promotion gate is intentionally blocked. The generated report records zero
label-based recall/F1 rather than presenting unverified performance as truth.
Before production use, domain experts must label a temporally held-out set,
validate geolocation and persistence rules, and evaluate image and tabular
models separately on representative regions and seasons.

## Reproduce

```powershell
.venv\Scripts\thermis.exe train-tabular
.venv\Scripts\thermis.exe evaluate-ranking
.venv\Scripts\python.exe scripts\verify_offline.py
```
