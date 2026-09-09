# Review 600 thermal cases — six exclusive CSV assignments

Supersedes #1's older suggested-label review queue. This replaces the review
workflow, not proof that a newer model or dataset already performs better.
Production promotion remains blocked pending credible independent evaluation.

Branch: **codex/retraining-upgrade**.
[Review guide](https://github.com/rajeet-04/SIH162/blob/codex/retraining-upgrade/reviews/thermal-v1/README.md)

Each reviewer owns **all 100 data rows** in their file (spreadsheet rows 2–101).
No review ID or sampled cell is assigned to another reviewer. Please edit only
your own CSV and open a PR against this branch mentioning #2.

- [ ] @NuclearVenom — [NuclearVenom.csv](https://github.com/rajeet-04/SIH162/blob/codex/retraining-upgrade/reviews/thermal-v1/NuclearVenom.csv)
- [ ] @rajeet-04 — [rajeet-04.csv](https://github.com/rajeet-04/SIH162/blob/codex/retraining-upgrade/reviews/thermal-v1/rajeet-04.csv)
- [ ] @DeepaliSingh10 — [DeepaliSingh10.csv](https://github.com/rajeet-04/SIH162/blob/codex/retraining-upgrade/reviews/thermal-v1/DeepaliSingh10.csv)
- [ ] @Somsubhra-Nandi — [Somsubhra-Nandi.csv](https://github.com/rajeet-04/SIH162/blob/codex/retraining-upgrade/reviews/thermal-v1/Somsubhra-Nandi.csv)
- [ ] @Subhra-Nandi — [Subhra-Nandi.csv](https://github.com/rajeet-04/SIH162/blob/codex/retraining-upgrade/reviews/thermal-v1/Subhra-Nandi.csv)
- [ ] @mishhverse — [mishhverse.csv](https://github.com/rajeet-04/SIH162/blob/codex/retraining-upgrade/reviews/thermal-v1/mishhverse.csv)

## How to review

Open the dated Worldview and OSM links, investigate the representative detection
and surrounding dates, and enter a supported label with source URL, evidence date,
evidence kind, confidence, notes, GitHub username and UTC review time. Set
`review_status=reviewed` when done. Use `uncertain` plus notes whenever evidence
is insufficient. Do not label routine facility heat as an accident. Never use a
model suggestion as independent evidence. Exact label definitions are in the guide.

All columns before `review_status` are frozen. Do not edit the reference index or
manifest. Download/save UTF-8 CSV with unchanged columns; no API keys or GPU needed.

## Scope and limitations

- 600 randomly sampled distinct 0.05-degree cell/month windows from historical
  FIRMS, not 600 confirmed events/sites. Adjacent cells can share a physical site.
- One reviewer per case, as requested: no duplicate work; agreement is not measured.
- No model predictions or suggested labels are shown.
- This is a **development-labeling batch**, not a pristine final ranking set.
  The newest 10% date range is excluded, but prior exploratory fits used some
  development dates. A separate independent evaluation cohort is still required.
- Uniform cell sampling does not guarantee enough industrial-fire examples.
  If a class lacks evidence, retain uncertainty and plan a targeted follow-up.

## Acceptance

- [ ] Six reviewer files submitted; all 600 cases reviewed or explicitly uncertain.
- [ ] `python -m thermis.review_packets --output reviews/thermal-v1 --validate` passes.
- [ ] Maintainer audits evidence provenance and adjudicates flagged conflicts.
- [ ] Class coverage and usable reference-label counts reported separately from weak labels.
- [ ] Separate follow-up tracks independent benchmark construction and model comparison;
      no automatic promotion based on completing this issue.
