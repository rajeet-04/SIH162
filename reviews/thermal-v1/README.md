# Thermal review batch v1 — six exclusive assignments

Review **only your own CSV**, all 100 data rows (spreadsheet rows 2–101).
Files are on branch `codex/retraining-upgrade`. No Python, GPU, API keys or local
satellite archives are needed to review. Download your CSV and open it in Excel,
LibreOffice or a spreadsheet editor, then save as UTF-8 CSV with the same columns.

| Reviewer | File | Work |
|---|---|---|
| @NuclearVenom | [NuclearVenom.csv](NuclearVenom.csv) | all 100 rows |
| @rajeet-04 | [rajeet-04.csv](rajeet-04.csv) | all 100 rows |
| @DeepaliSingh10 | [DeepaliSingh10.csv](DeepaliSingh10.csv) | all 100 rows |
| @Somsubhra-Nandi | [Somsubhra-Nandi.csv](Somsubhra-Nandi.csv) | all 100 rows |
| @Subhra-Nandi | [Subhra-Nandi.csv](Subhra-Nandi.csv) | all 100 rows |
| @mishhverse | [mishhverse.csv](mishhverse.csv) | all 100 rows |

## What a row means

One sampled 0.05-degree cell and one observed calendar month, with a representative
FIRMS detection and summary of detections in that cell/month. This is a review
window, **not a proven physical fire event or facility**. Adjacent cells can refer
to the same physical site; multiple sources can coexist within a cell. Assess the
representative detection and document conflicts rather than assigning every
monthly detection the same class. Labels must not automatically propagate to all
observations at the cell or facility.

600 distinct cells were sampled uniformly without replacement, seed 26162; an
observed month was then sampled uniformly per cell. This is not anomaly-ranked,
class-balanced, or representative of detection-level prevalence. Model predictions,
provider type flags and suggested labels are deliberately absent. The sample is
from the India-region bounding box, which includes neighboring countries.

**Development labels only.** All observations at/after 2025-05-18 08:12 UTC were
excluded. Some earlier dates were used in exploratory training/validation, so this
batch is not an untouched test benchmark. Reference-label provenance still needs
auditing before supervised use. One reviewer per case does not measure agreement.

## Review procedure

1. Open `worldview_url`. It centers the representative location and date; add
   appropriate VIIRS/Sentinel/Landsat imagery and thermal layers in Worldview as
   available. Check nearby dates across `window_month`. Clouds, coarse resolution,
   missing coverage or absent detections are not proof of no fire. These links
   are browsing starting points, not pre-verified imagery evidence.
2. Open `osm_url` to investigate facility and land-cover context. Current OSM is
   not proof that a facility existed on the historical date. Record the actual
   evidence you find, including its date. Search official incident reports or
   provider-backed imagery where needed. A hotspot near a factory alone does not
   prove an industrial accident. FRP is fire radiative power in MW, not temperature.
3. Fill the editable columns below. If a window contains multiple plausible
   sources, or you cannot support a class, choose `uncertain` and explain why.
4. Set `review_status=reviewed` only when finished. Save only your file; submit a
   pull request targeting `codex/retraining-upgrade` (or commit directly if your
   team uses that workflow). Mention issue #2 and your username. Never edit another
   reviewer's file, `reference_index.csv`, or `manifest.json`.

## Editable columns (everything else is immutable)

| Column | Allowed content |
|---|---|
| review_status | `pending` or `reviewed` |
| label | one value from the definitions below |
| evidence_url | direct public source URL; multiple URLs may be separated by `; ` |
| evidence_date | evidence observation/incident date, ISO `YYYY-MM-DD`; not today's access date |
| evidence_kind | e.g. `dated_satellite_imagery`, `official_incident_report`, `facility_inventory`, `multiple_sources` |
| confidence | `low`, `medium`, `high`; subjective review confidence, not model probability |
| notes | why the evidence supports the label; dates, facility name, conflicts, cloud limitations |
| reviewed_by | your exact assigned GitHub username |
| reviewed_at_utc | completion time, e.g. `2026-09-09T12:00:00Z` |

For `uncertain`, notes are mandatory; add identity/time and other evidence when
available. For every other completed label, all evidence, confidence, identity and
completion-time fields are mandatory. Do not paste credentials or private data.

## Label definitions

- `persistent_industrial_heat_or_flare`: evidence identifies industrial heat/flare
  context and repeated activity compatible with routine operation. Repetition
  alone is insufficient; non-industrial recurrent sources exist.
- `industrial_fire_candidate`: evidence supports an episodic fire at industrial
  infrastructure, rather than routine heat. Include dated incident evidence or
  convincing imagery/context. This label does not certify an accident.
- `vegetation_fire_candidate`: evidence supports burning vegetation outside a
  supported agricultural-burn interpretation. Thermal activity in a forest pixel
  alone may remain uncertain.
- `agricultural_burn_candidate`: dated evidence supports burning crop/residue at
  agricultural land. Cropland presence alone is insufficient.
- `uncertain`: absent, ambiguous, mixed or conflicting evidence; other source
  types not covered above. This is a useful answer, not a failed review.

Do not invent minority-class examples to balance the dataset. If most rows remain
uncertain, report that result; it determines the next targeted evidence search.
An AI-generated explanation is not independent evidence. Never copy old model or
rule labels into these columns without investigation.

## Maintainer validation and reproducibility

After installing project dependencies, from the branch root:

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m thermis.review_packets --output reviews/thermal-v1 --validate
```

The validator checks IDs, exact ownership, immutable reference fields, allowed
labels and required evidence fields. It cannot verify the truth of the evidence.
CSV parsers should preserve literal text; do not execute spreadsheet formulas.

`manifest.json` records source hashes, sampling seed, date exclusion and initial
CSV hashes. Reviewer-file hashes naturally change after edits; the reference
index remains frozen. The generator refuses to overwrite an existing directory.
To reproduce into a NEW directory (requires the downloaded annual archives):

```powershell
python -m thermis.review_packets --observations data/cleaned/firms-snpp-2024-v1/observations.parquet data/cleaned/firms-snpp-2025-v1/observations.parquet --output reviews/thermal-v1-reproduced --reviewers NuclearVenom rajeet-04 DeepaliSingh10 Somsubhra-Nandi Subhra-Nandi mishhverse
```

Review completion does not automatically promote a model. Next: audit evidence,
adjudicate conflicts, check class coverage, train with reference/weak labels kept
separate, and build a genuinely independent event/site evaluation cohort.
