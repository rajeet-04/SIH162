# SIH26162 Phased Execution Roadmap

**Branch:** `feature/sih26162-mvp`
**Worktree:** `.worktrees/sih26162-mvp`
**Detailed plan:** `docs/superpowers/plans/2026-09-01-sih26162-mvp-model.md`
**Design:** `docs/superpowers/specs/2026-09-01-sih26162-mvp-model-design.md`

## Phase Sequence

| Phase | Tasks | Deliverable | Mandatory checkpoint |
| --- | ---: | --- | --- |
| A. Runtime and Source Trust | 1-4 | Locked runtime and trusted source manifests | Checksums, readability, duplicates, schemas, and join coverage pass |
| B. Labels, Splits, Features | 5-7 | Training-ready event table | Label provenance and automated leakage tests pass |
| C. Primary Decision Model | 8 | Calibrated hierarchical tabular model | Internal validation and calibration reviewed; ranking remains sealed |
| D. Image and Fusion | 9-10 | Image verifier and uncertainty-aware risk | GPU/CPU portability and disagreement behavior pass |
| E. Service and Product | 11-12 | Offline API and judge-facing dashboard | Offline cases, API tests, frontend tests, and core journey pass |
| F. Final Evaluation | 13-14 | Ranking report and verified MVP handoff | Frozen ranking evaluation, promotion decision, CI, and offline rehearsal pass |

## Execution Rule

Complete phases in order. Stop at every phase exit gate, record test output and artifact hashes, and review the evidence before starting the next phase. Phase F is one-way: opening the newest 10% ranking set is allowed only after all model and policy artifacts are frozen.

## Phase Checkpoint Record

Each checkpoint records:

- Commit hash.
- Commands executed and pass/fail status.
- Generated artifact paths and SHA-256 hashes.
- Data exclusions or unresolved quality issues.
- Metric summary and decision to continue, revise, or stop.
- Confirmation that raw external datasets were not modified.
