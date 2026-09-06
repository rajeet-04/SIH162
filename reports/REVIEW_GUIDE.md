# Reviewer guide: completing the authoritative evaluation

## Why you are needed
The tabular scores in the dashboard measure decision-logic reproduction, not
field accuracy (see model card). Only human-confirmed labels on the sealed
ranking set can prove real performance. Budget: 300+ confirmed rows.

## Steps (about 2 hours for 300 rows)
1. Open `reports/review_queue_suggested.csv`.
2. Work the `split == ranking` rows with a non-blank `suggested_stage_2` first.
3. For each row: open `worldview_url` (NASA Worldview, thermal-anomaly layers
   pre-selected at the event date), check `suggestion_rationale` and the
   FRP/persistence/flare columns, then accept or correct into
   `expert_label_stage_1` / `expert_label_stage_2` using these values:
   - stage 1: `emergency_or_transient_fire`, `persistent_or_non_emergency_heat`, `uncertain`
   - stage 2: `industrial_fire`, `persistent_industrial_heat_or_flare`,
     `wildfire_or_agricultural_burn`, `other_or_uncertain`
4. Save (keep the header), then run:
   `.\.venv\Scripts\thermis.exe evaluate-ranking --labels-csv reports/review_queue_suggested.csv --label-column expert_label_stage_2`
5. Read `reports/ranking/ranking_expert_report.json`: promotion needs
   industrial recall ≥ 0.85, macro F1 ≥ 0.75, persistent false-alert ≤ 0.15,
   and ≥ 300 labeled rows — otherwise the gate stays blocked with reasons.

## What NOT to do
- Do not copy `suggested_*` into `expert_*` unread; suggestions are weak.
- Do not label from the model's predicted class; use the satellite evidence.
- Do not edit any other file; the pipeline re-reads only this CSV.
