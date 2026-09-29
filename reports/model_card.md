# THERMIS SIH26162 model card

## Intended use

THERMIS is an offline-capable decision-support MVP for detecting and triaging
industrial fires and persistent thermal sources. It ranks an incoming thermal
event for analyst review; it is not an autonomous emergency or regulatory
decision system.

## Data and labels

- Canonical event table: 12,165 accepted rows (9,186 archival + 2,979 live FIRMS
  India detections, of which 2,860 fall in the India window); 6 malformed rows
  were rejected.
- Supporting sources are tracked in `config/sources.yaml` and the generated
  manifests. Gas-flaring facility points are used for proximity features.
- Labels are conservative, rule-derived candidates using persistence and
  industrial/flare proximity. 10,847 rows remain review-required and 1,318 are
  eligible for supervised training.
- The image verifier uses 47,992 readable Fire/No_Fire images from the local
  Training and Test folders. The official Test set and a deterministic grouped
  10% holdout are excluded from training, leaving 35,437 development images and
  12,555 ranking rows. It is a ResNet18 verifier, not yet a production
  satellite-image classifier.
- The 288.85GB MiCASA 3-hourly/daily archive is environmental flux data with
  no fire-class label alignment to the prepared event period. It is retained
  for a future environmental-context model and was not incorrectly consumed as
  fire-classifier training data.

## Training and leakage controls

The newest 10% of whole event time groups is reserved as the untouched ranking
set: 10,948 development rows and 1,217 ranking rows. Ranking rows are excluded
from fitting, calibration, threshold selection, and demo construction. Inside
development, the tabular estimator fits on the oldest 80% of eligible rows and
uses the later 20% for sigmoid calibration diagnostics. Scene-group isolation
is enforced for image records.

## Current artifacts

Stage 1 and Stage 2 are calibrated CatBoost classifiers on the exp branch,
retrained under an honest protocol: `nearest_industrial_distance_m` removed
from model inputs (it duplicated flare distance exactly), and a chronological
three-way split — 894 fit / 192 calibration / 192 validation rows, metrics from
the untouched validation slice only, `ranking_rows_used=0`. Ablation
(`reports/baseline_comparison.json`): HGB without the three rule-read features
scores 0.56 macro-F1 vs 0.99+ with them, so the headline score measures
decision-logic reproduction, not field accuracy. Do not present it as
real-world performance; promotion stays blocked pending authoritative labels.
Baseline detail: logreg 0.75 vs RF/HGB/CatBoost 1.0 macro-F1 — tree models tie
at the ceiling because labels are rule-derived, so CatBoost is retained for
calibrated probabilities, not an accuracy edge. The shipped image
verifier is a ResNet18 trained with grouped dev-val early stopping (31,893 fit
+ 3,544 dev-val rows, batch 320, AMP, augmentation, weight decay 1e-4 on the
RTX 5050; best epoch 4 of 8, dev-val NLL 0.86) and exported as CPU-loadable
TorchScript. A prior 12-epoch unvalidated checkpoint memorized development
(dev NLL ~1e-5) and was rejected despite a higher raw holdout score, because it
was selected without validation. An experimental 12-band U-Net segmentation
baseline exists under `models/seg-baseline/` (scene1+2 fit, scene3 dev-val,
scene4 holdout) but is not wired into the demo: fire pixels are 1.6% of the
data and the smoke runs collapsed to background/all-fire extremes.

## Evaluation and limitations

Sealed image-holdout results for the shipped verifier (`reports/image_holdout.json`,
all 12,555 rows, never trained on): accuracy 0.665, fire recall 0.78, no-fire
recall 0.39, mean confidence 0.91, 10-bin ECE 0.25. Temperature scaling fit on
development only returned T~1.0 (no-op), confirming overconfidence comes from
overfitting rather than a correctable calibration shift. The verifier therefore
remains supporting evidence only: when it disagrees with the tabular model the
fusion layer reports `uncertain / review required` instead of a forced class.
The tabular ranking set has no authoritative labels in the available sources, so
the promotion gate is intentionally blocked. The generated report records zero
label-based recall/F1 rather than presenting unverified performance as truth.
Before production use, domain experts must label a temporally held-out set,
validate geolocation and persistence rules, and evaluate image and tabular
models separately on representative regions and seasons.

## Reproduce

```powershell
.venv\Scripts\thermis.exe train-tabular
.venv\Scripts\thermis.exe train-image --epochs 8 --batch-size 320 --max-records 0 --val-fraction 0.1 --weight-decay 0.0001 --augment
.venv\Scripts\thermis.exe evaluate-ranking
.venv\Scripts\python.exe scripts\verify_offline.py
.venv\Scripts\python.exe scripts\evaluate_image_holdout.py
.venv\Scripts\python.exe scripts\train_seg_baseline.py
```
