# SIH26162 MVP and Model Design

Date: 2026-09-01  
Status: Approved design  
Project: AI-based detection and classification of industrial fires and persistent thermal sources

## 1. Objective

Build a defensible MVP that accepts a satellite thermal anomaly, distinguishes an actionable fire from persistent or non-emergency heat, classifies the likely source, calculates a calibrated risk score, and presents the supporting geospatial and temporal evidence to an authority.

The MVP must prioritize model quality, explainability, offline demo reliability, and measurable performance over breadth. Mobile applications, production notification delivery, IoT sensors, drone integration, end-to-end multimodal neural networks, and predictive fire-spread simulation are outside the first release.

## 2. Success Criteria

The MVP is successful when it can:

1. Ingest and normalize representative FIRMS-style thermal events.
2. Enrich each event with historical, environmental, burned-area, and industrial context.
3. Produce a two-stage classification with calibrated probabilities.
4. Mark conflicting or out-of-distribution evidence as uncertain instead of forcing a class.
5. Explain the prediction with model evidence and a 7/30/90-day history.
6. Demonstrate performance on an untouched chronological and grouped 10% ranking set.
7. Run a frozen offline demonstration without depending on external services.

Target promotion thresholds on the final ranking set are:

- Industrial-fire recall at least 85%.
- Macro F1 at least 0.75.
- Persistent-source false-alert rate at most 15%.
- No severe collapse across major geographic or temporal segments.
- CPU inference below one second per event.
- Confidence probabilities that track observed accuracy closely enough to support risk decisions.

## 3. Approved Model Strategy

Use a tabular-first hierarchical decision engine with a separate satellite-image verifier.

### 3.1 Stage 1

Predict one of:

- `active_or_transient_fire`
- `persistent_or_non_emergency_heat`

This stage controls alert urgency and is optimized for actionable-fire recall while limiting persistent-source false alerts.

### 3.2 Stage 2

Predict one of:

- `industrial_fire`
- `persistent_industrial_heat_or_flare`
- `wildfire_or_agricultural_burn`
- `other_or_uncertain`

Stage 2 provides source attribution. It does not override Stage 1 urgency without going through the calibrated decision layer.

### 3.3 Image Verifier

The verifier supplies supporting evidence from a satellite patch. It cannot silently override the tabular decision engine. When the verifier and tabular model disagree materially, the decision layer emits `uncertain / review required` and exposes both probability distributions.

The first image baseline uses transfer learning with EfficientNet-B0 or ConvNeXt-Tiny. U-Net segmentation is added only after mask alignment, class balance, and scene provenance pass quality checks. Training uses mixed precision and moderate image sizes compatible with the available NVIDIA GeForce RTX 5050 and approximately 8 GB VRAM.

### 3.4 Future Model Work

An end-to-end multimodal neural network is deferred until the tabular and image components are independently validated. It is not required for the MVP.

## 4. Available Data Sources

### 4.1 Structured and Image Sources in `D:\data`

- `csv_output/sentinel_frp_combined.csv`: 4,596 structured FRP records.
- Regional FRP files for the Amazon basin, Central Africa, China industrial areas, Gulf flaring, and Siberian wildfire cases.
- `flaresat_github/dataset/flare_dataset.csv`: 7,337 flare examples.
- `flaresat_github/dataset/fire_dataset.csv`: 265 fire examples.
- `flaresat_github/dataset/urban_dataset.csv`: 263 urban examples.
- `flaresat_github/dataset/metadata/flare_dataset_metadata.csv`: 10,409 flare metadata records.
- FireSat and FlareSat patch and mask archives.
- 2,466 `.npz` satellite patches across `scene1` through `scene4`.
- 39,375 training JPG images and 8,617 test JPG images.
- One Sentinel-2 Level-1C SAFE scene.
- Existing Xception model definition and weights, retained as a compatibility-tested baseline rather than accepted as the production model.

### 4.2 Environmental and Burned-Area Sources

- `R:\MiCASA_FLUX`: monthly MiCASA NetCDF files.
- `R:\MiCASA_FLUX_DAILY`: daily MiCASA NetCDF files.
- `R:\MiCASA_FLUX_3H`: 3-hourly observations packaged as daily NetCDF files.
- `D:\data\MCD64A1` and `D:\MCD64A1`: MODIS burned-area HDF files. Their relationship must be resolved by content manifest before either copy is used.

### 4.3 Required Context Sources

- NASA FIRMS-style event records for thermal detections and operational inference.
- OpenStreetMap industrial land use and facility context.
- Known gas-flare point metadata from FlareSat sources.

External enrichment is cached. The MVP includes a frozen representative subset so the demonstration remains available offline.

## 5. Data Architecture

Raw datasets remain immutable. Derived artifacts are stored separately and identified by manifest and schema versions.

```text
raw sources
  -> file and source manifests
  -> data-quality gate
  -> normalized source tables
  -> canonical thermal-event table
  -> time-safe feature tables
  -> label table with provenance and confidence
  -> split manifests
  -> trained models and evaluation artifacts
```

Recommended project layout:

```text
data/
  manifests/
  cleaned/
  features/
  labels/
  splits/
models/
reports/
src/
  inventory/
  cleaning/
  features/
  labeling/
  training/
  evaluation/
  inference/
app/
tests/
```

Large raw data remains in its current external drives. Project manifests reference absolute source paths while portable demo artifacts use project-relative paths.

## 6. Data Quality Gate

Every source file receives a manifest entry containing:

- Absolute source path.
- Source family and expected role.
- File size and modification timestamp.
- Fast checksum and, for duplicate candidates, SHA-256.
- Date or scene identifier parsed from the filename or metadata.
- Readability status.
- Dimensions, variables, channels, and dtypes where applicable.
- Quality flags and exclusion reason.

Exclude from training:

- Corrupt, truncated, zero-byte, or unreadable files.
- Files missing required variables or valid dimensions.
- Duplicate content hashes.
- Invalid timestamps or coordinates.
- Observations with impossible physical values after units are confirmed.
- Images whose masks cannot be aligned to their source patch.
- Records with unknown provenance when they could cross split boundaries.

Incomplete downloads remain quarantined. Failed-download URL logs remain preserved as recovery metadata.

## 7. Canonical Event Contract

The unit of analysis is one thermal detection at one location and timestamp. A unique `event_id` represents this grain.

Minimum fields are:

```text
event_id
timestamp_utc
latitude
longitude
brightness_temperature
frp
frp_uncertainty
day_night
cloud_indicator
water_indicator
window_indicator
micasa_environmental_features
modis_burned_area_features
nearest_industrial_distance_m
nearest_flare_distance_m
persistence_7d
persistence_30d
persistence_90d
spatial_cluster_size
spread_or_movement_rate
source_dataset
label_stage_1
label_stage_2
label_source
label_confidence
quality_flags
```

All timestamps are normalized to UTC and coordinates to latitude `[-90, 90]` and longitude `[-180, 180]`. Optional enrichment remains null when unavailable; critical FIRMS coordinates, event time, and thermal measurements are required.

## 8. Feature Design

Feature families are:

1. Thermal intensity: brightness temperature, FRP, uncertainty, radiance, channel statistics, and local thermal contrast.
2. Temporal persistence: prior detections within 7, 30, and 90 days at location and cluster levels.
3. Spatial behavior: cluster size, movement, spread, and distance to industrial facilities, flare points, vegetation, and urban areas.
4. Environmental context: MiCASA weather and flux variables sampled at or before event time.
5. Burned-area context: MODIS overlap, prior burned-area history, and vegetation-region context.
6. Observation context: day/night, cloud, water, window, viewing angle, and acquisition-quality indicators.
7. Image evidence: independently calibrated image-verifier probabilities and image-quality flags.

No feature may use information recorded after the prediction timestamp. Historical aggregates are computed with strictly backward-looking windows.

## 9. Label Design

Labels retain their source and confidence. Weak supervision is acceptable for bootstrapping but is not treated as ground truth.

- Repeated stationary detections aligned with known flare points support `persistent_industrial_heat_or_flare`.
- Burned-area overlap and moving or spreading vegetation-region events support `wildfire_or_agricultural_burn`.
- Sudden, non-persistent high-FRP events near industrial infrastructure support industrial-fire candidates, subject to manual verification or authoritative event evidence.
- Urban and no-fire sources support image-background verification, not event-level non-emergency labels without temporal context.
- Conflicting, weak, or ambiguous evidence receives `other_or_uncertain` or is excluded from supervised training.

The general `fire` label is never automatically converted to `industrial_fire`; geographic and temporal context is required.

## 10. Train, Validation, and Ranking Split

The newest 10% is the untouched ranking set. The older 90% contains model training and internal validation.

The split is chronological and grouped:

- Patches from the same source satellite scene remain in one split.
- Records from the same original acquisition or archive remain in one split.
- Facility and tight location clusters remain in one split where practical.
- Preprocessing, imputation, feature selection, class weighting, and calibration are fit without the ranking set.
- The ranking set is evaluated only after model architecture, features, hyperparameters, calibration method, and thresholds are frozen.

If a single split leaves important regions or classes absent, use grouped temporal cross-validation inside the older 90% for development, while preserving the final newest 10% unchanged.

## 11. Training Protocol

### 11.1 Baselines

Build a deterministic rules baseline and simple logistic-regression baseline before boosted trees. This proves whether the trained model adds measurable value.

### 11.2 Candidate Models

Compare Random Forest, CatBoost, and LightGBM for Stage 1. Compare CatBoost and LightGBM for Stage 2. Use class weights and decision thresholds selected from internal validation. Do not use indiscriminate random oversampling that destroys temporal or geographic structure.

### 11.3 Calibration

Compare Platt scaling and isotonic regression on internal validation data. Select the calibration method using reliability curves, expected calibration error, Brier score, and class-wise behavior.

### 11.4 Image Training

Split image data by source scene, acquisition, and geography. Compare transfer-learning classifiers first. Introduce segmentation only when masks pass alignment checks. Preserve the existing test image folder as an isolated evaluation source until provenance and class definitions are verified.

### 11.5 Fusion

Fuse calibrated tabular and image probabilities through an explicit weighted or meta-classifier layer trained on internal validation predictions. Record the contribution of each component. Material disagreement produces uncertainty rather than a forced class.

## 12. Evaluation and Promotion

Stage 1 metrics:

- Actionable-fire recall.
- Persistent-source specificity and false-alert rate.
- PR-AUC.
- Balanced accuracy.
- Brier score and expected calibration error.

Stage 2 metrics:

- Macro F1.
- Industrial-fire recall.
- Per-class precision and recall.
- Balanced accuracy.
- Log loss.
- Brier score and expected calibration error.

The model ranking score is:

```text
30% industrial-fire recall
20% macro F1
15% persistent-source specificity
15% PR-AUC
10% calibration quality
10% inference speed
```

In addition to global metrics, report performance by source dataset, geography, time period, day/night, confidence band, and missing-feature pattern. A model is not promoted when aggregate performance hides severe segment collapse.

## 13. Prediction and Risk Contract

Every prediction returns:

```text
event_id
stage_1_class and probabilities
stage_2_class and probabilities
image_verifier probabilities when available
final class
risk score from 0 to 100
risk band
uncertainty and incomplete-context flags
top evidence features
model version
feature timestamp
source provenance
```

Risk is a policy layer separate from raw class probability. It combines calibrated actionable-fire probability, industrial-source probability, thermal severity, nearby infrastructure exposure, context completeness, and uncertainty. Risk thresholds are frozen from internal validation before final ranking evaluation.

## 14. MVP Product Flow

### 14.1 National Thermal-Event Map

Show recent thermal events colored by predicted class and sized by risk. Provide time, region, risk, confidence, and source filters. Summary cards show total detections, events requiring review, high-risk industrial candidates, and estimated false alerts suppressed.

### 14.2 Event Investigation Panel

Selecting an event shows classification, calibrated confidence, risk, time, nearby industrial context, historical persistence, FRP change, burned-area context, image evidence, missing-context warnings, and the most influential evidence.

### 14.3 Time Machine

The 7/30/90-day timeline visually separates stationary persistent heat, moving or spreading wildfire, and sudden industrial anomalies.

### 14.4 Model Evaluation

Show the untouched ranking-set confusion matrix, industrial-fire recall, false-alert reduction, calibration, performance by segment, and selected errors. Metrics display the data cutoff and model version.

The core journey is:

```text
map -> select anomaly -> inspect evidence -> replay history -> decide whether to escalate
```

Alerts are simulated in the MVP. Production SMS, email, mobile applications, and emergency dispatch are future integrations.

## 15. System Components

1. Inventory service: builds source manifests and quality status.
2. Cleaning and normalization pipeline: creates canonical source tables.
3. Feature pipeline: performs time-safe geospatial and temporal enrichment.
4. Label pipeline: creates provenance-aware labels and review queues.
5. Training pipeline: trains, calibrates, evaluates, and versions models.
6. Inference service: exposes event classification, risk, and evidence.
7. Geospatial application: provides the map, investigation, time machine, and evaluation views.
8. Offline demo package: contains frozen events, features, images, and predictions.

The first implementation may use Parquet for analytical artifacts, FastAPI for inference, and a lightweight geospatial web client. PostgreSQL/PostGIS is introduced only if the MVP query volume or spatial workflows require it.

## 16. Failure Handling

- Unreadable source files are quarantined and excluded with a manifest reason.
- Missing optional features remain null and reduce confidence or add `incomplete_context`.
- Missing critical event identifiers, coordinates, timestamps, or thermal measurements reject the event.
- Enrichment-service failures do not block ingestion; cached or missing values are recorded.
- Out-of-distribution inputs produce `uncertain / review required`.
- Model disagreement produces uncertainty and exposes both evidence paths.
- Every prediction is auditable by model version, feature timestamp, source provenance, and probabilities.

## 17. Reproducibility

Each training run records:

- Data-manifest hash.
- Split manifest.
- Feature schema version.
- Label schema version.
- Model and calibration configuration.
- Random seeds.
- Software and CUDA environment.
- Metrics and segment diagnostics.
- Model artifacts and decision thresholds.

## 18. Testing

Testing includes:

- Unit tests for timestamps, coordinates, temporal windows, distances, and label rules.
- Data-contract tests for schema, ranges, enums, uniqueness, readability, and freshness.
- Leakage tests for future information, scene overlap, acquisition overlap, and grouped entities.
- Model tests for promotion thresholds, calibration, class collapse, and segment degradation.
- API tests for valid, incomplete, uncertain, out-of-distribution, and malformed events.
- End-to-end tests from stored event through enrichment, inference, and dashboard evidence.
- GPU training and CPU inference verification.
- Offline demo rehearsal with network access unavailable.

## 19. Implementation Phases

### Phase 0: Runtime and Repository Foundation

Create a project-local `uv` environment, lock dependencies, establish manifests and configuration, and verify CUDA-compatible training libraries.

### Phase 1: Data Audit and Canonicalization

Profile every selected source, resolve duplicate MCD64A1 locations, inspect CSV and NPZ schemas, validate image-mask alignment, create manifests, and define exclusion rules.

### Phase 2: Event Table and Labels

Normalize structured events, add source provenance, implement canonical labels and confidence, and produce the chronological grouped split manifests.

### Phase 3: Tabular Baseline

Implement time-safe feature engineering, train rules and statistical baselines, train Stage 1 and Stage 2 boosted trees, calibrate probabilities, and freeze ranking criteria.

### Phase 4: Image Verification

Train the scene-grouped image baseline, evaluate independently, add segmentation only if justified, and implement transparent fusion and disagreement handling.

### Phase 5: Inference and MVP Product

Implement the inference contract, risk policy, evidence generation, map, event panel, time machine, evaluation view, and frozen offline demo.

### Phase 6: Final Validation

Run the untouched 10% ranking evaluation, segment diagnostics, CPU inference checks, end-to-end tests, and offline presentation rehearsal.

## 20. Explicit Non-Goals for MVP

- Production emergency dispatch.
- Mobile application.
- Live SMS or email delivery.
- IoT or drone integration.
- Predictive fire-spread simulation.
- Global production-scale ingestion.
- Opaque end-to-end multimodal modeling.
- Claims of identifying an industrial accident without authoritative or sufficiently strong corroborating evidence.

## 21. Key Risks and Mitigations

1. Weak industrial-fire ground truth: retain provenance and confidence, create a review queue, and report results separately for strong and weak labels.
2. Scene and geographic leakage: enforce grouped chronological splits and automated overlap tests.
3. Class imbalance: use class weights, calibrated thresholds, macro metrics, and per-class reporting.
4. Heterogeneous source semantics: maintain a canonical label map and never concatenate sources without explicit transformation.
5. External-service failure during demo: use cached enrichment and a frozen offline case set.
6. Attractive but unreliable confidence: calibrate probabilities and expose uncertainty and missing context.
7. Aggregate metrics hiding failures: require segment diagnostics before promotion.

## 22. Final Design Decision

The MVP is a geospatial thermal-event reasoning system, not merely a fire-image detector. Its primary value is the combination of time-safe event classification, persistent-source suppression, calibrated risk, and visible evidence. The tabular hierarchical model is the operational decision engine; satellite imagery verifies and enriches that decision.
