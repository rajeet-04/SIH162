# SIH26162 retraining upgrade

## Global constraints
Preserve deployed models and original datasets. CUDA training must fit 8 GB VRAM.
Newest 10% is sealed final evaluation; model selection uses older data only.
No invented labels or claims that weak-label agreement measures accident accuracy.
Missing imagery/context must be explicit, never silently synthesized.

### Task 1: audit and examples (controller)
Versioned checksummed source audit, logical duplicates, invalid archive exclusions,
explicit imagery admission gate. Build FIRMS event examples with past-only history,
separate facility/flare sources, provenance-aware labels and event/site grouping.
Retrospective burned-area evidence is never an online feature. MiCASA deferred.

### Task 2: CUDA challenger training (worker)
Own only src/thermis/retrain_models.py and tests/test_retrain_models.py.
Implement train_challengers(frame, output, feature_columns, epochs=20, batch_size=256,
device='auto', resume=False) returning a JSON-serializable run report. Inputs have
timestamp_utc, event_id, site_id, episode_id, region_id, split (train/validation/test/purged),
target, label_tier (reference/weak/unknown), and numeric features.
There may be unknown labels; never train on them. Input test rows are never used in
fitting, scaling, checkpoint selection or reported selection metrics. Training requires
at least two represented classes; fail closed with explicit report if not available.
Train CUDA CatBoost baseline and compact PyTorch GRU using preceding observations
from the same site (max 16, with strict timestamp ordering), plus current numeric
features through a small head. Fit imputation/scaling on train only, encode NaNs with
missingness flags. Dataset constructs sequences lazily, tensors batch to GPU only.
Use AMP on CUDA, gradient accumulation (effective batch >=256), deterministic seed,
OOM batch backoff with restart from epoch checkpoint (do not keep partial updates),
resume checkpoints with model/optimizer/scaler/RNG state and dataset/config identity.
Save model artifacts under output only, never models/tabular or deployed locations.
Support CPU artifact loading/predict via load_challenger(path, device='cpu') with
predict_proba(frame) and classes_; preserve preprocessing and class mapping.
Metrics on identical validation rows: macro F1, per-class precision/recall/support and
PR AUC, ECE, event-level persistent false industrial alerts. Separate reference and
weak strata; unsupported metrics null. Report unseen-site/region validation subsets.
No automatic promotion; absent adequate independent industrial labels is an explicit
block. Save dependency versions, feature schema, checkpoint and model card.
Test first: future/test rows cannot affect fit/sequence, checkpoint CPU roundtrip,
missing features and unknown labels, insufficient classes, metrics null when unsupported,
resume identity mismatch and bounded OOM recovery. Real tiny CPU runs suffice for tests;
controller runs actual CUDA workload after integration. Do not consume real sealed test.

### Task 3: integration and execution (controller)
CLI workflow audit/prepare/train, safe run directories, source recovery attempt from
original provider, actual CUDA training on eligible real data, end-to-end reload test,
documentation of completed/blocked work and no unearned model promotion.

### Task 4: independent review
Review data leakage, artifact safety and scientific validity. Fix important findings.
