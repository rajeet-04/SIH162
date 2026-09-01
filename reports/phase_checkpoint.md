# SIH26162 phase checkpoint

Branch: `feature/sih26162-mvp`  
Latest implementation checkpoint: see `git log --oneline` for the current HEAD.

## Exit evidence

| Phase | Result | Evidence |
|---|---|---|
| A. Runtime and source trust | complete | 10 configured roots validated; Global Fire Atlas checksums and source manifests were checked; raw roots were not modified |
| B. Labels, splits, features | complete | 9,186 accepted events; 6 rejects; 8,267 development / 919 ranking rows; historical leakage tests pass |
| C. Primary decision model | complete | calibrated Stage 1/2 CatBoost bundles; 739 fit + 185 calibration rows per stage; ranking rows used = 0 |
| D. Image and fusion | complete | CUDA Torch 2.13.0+cu130 on RTX 5050; full 35,437-image development train with batch 128/AMP; grouped holdout 12,555; CPU TorchScript export; fusion tests pass |
| E. Service and product | complete | FastAPI health/events/metrics/predict rehearsal passes; Bun/Vite test and production build pass |
| F. Final evaluation and handoff | complete with promotion blocked | frozen ranking report generated; promotion is blocked because ranking labels are unavailable; model card, data dictionary, CI, and offline rehearsal are present |

## Verification snapshot

- Python: `32 passed`; Ruff clean.
- Frontend: `1` focused Vitest test file passed; Vite production build passed.
- Browser: live Playwright CLI rehearsal passed map selection, evidence, 7D
  timeline, and evaluation view.
- Offline rehearsal: API status `200` for health, events, metrics, and predict;
  CPU image verifier output shape `(1, 2)`.
- GPU: `torch.cuda.is_available() == True`; device `NVIDIA GeForce RTX 5050
  Laptop GPU`; CUDA `13.0`.
- Ranking: 919 rows, no authoritative labels, `promoted=false`.

## Required next step before claiming production accuracy

Obtain expert labels for the sealed ranking set (or a newly collected temporal
holdout), then rerun `thermis evaluate-ranking`. Do not interpret the current
development diagnostics or zero-label ranking metrics as field accuracy.
