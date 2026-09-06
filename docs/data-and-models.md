# Data and model setup guide

This guide separates what a fresh checkout already contains from what must be
obtained locally. It is intentionally platform-neutral: use `/` in commands
on macOS/Linux and PowerShell paths only when working on Windows.

## Fresh checkout inventory

| Item | In Git | Offline demo | Retraining |
|---|---:|---:|---:|
| Python source and tests | yes | yes | yes |
| Bun dashboard source | yes | yes | no |
| Demo events | yes | yes | no |
| Tabular model bundles | yes | yes | generated if retraining |
| Image TorchScript verifier | yes | `/verify-image` | generated if retraining |
| Prepared tabular data | yes | verification | yes |
| Raw satellite/image archives | no | no | yes |
| Live SQLite database | no | no | runtime only |
| FIRMS/NASA credentials | no | no | live mode only |

The checked-in model hashes can be recorded with `git ls-files models | xargs
shasum -a 256` on macOS/Linux, or `Get-ChildItem models -Recurse -File |
Get-FileHash -Algorithm SHA256` in PowerShell.

## Raw source layout

`config/sources.yaml` is local-machine configuration; its current Windows drive
letters are not portable assumptions. Replace each value with an absolute or
repository-relative path on the new machine:

| Key | Contents | Role |
|---|---|---|
| `structured` | FIRMS/structured CSV inputs | event preparation |
| `firesat` | FireSat image folders | image manifest |
| `flaresat` | flare facility metadata | contextual proximity |
| `mcd64a1` | MODIS burned area | future label/context enrichment |
| `global_fire_atlas` | ignition/perimeter archives | future label/context enrichment |
| `micasa_3h`, `micasa_daily` | environmental flux NetCDF | optional context, not current fire labels |

Do not put raw archives or API keys inside the Git worktree unless ignored.
The pipeline writes derived artifacts under `data/`, `models/`, and `reports/`.

## Rebuild order

1. `inventory` records files, sizes, identities, and readability.
2. `prepare-manifests` creates image/environment/Fire Atlas manifests.
3. `prepare-events` cleans and deduplicates structured events.
4. `build-features` creates the fixed tabular feature contract.
5. `make-splits` creates the chronological 90% development / 10% ranking split.
6. `train-tabular` fits only development rows.
7. `train-image` fits only grouped development images and exports TorchScript.
8. `evaluate-ranking` scores the sealed ranking partition without promotion.

The promotion gate remains blocked until domain experts provide authoritative
labels for the final ranking set. Rule-derived labels can make tree models
appear perfect without measuring real-world accuracy.

## Platform notes

- Windows/NVIDIA: `uv sync` selects the CUDA 13.0 PyTorch wheels specified by
  the project. Confirm `torch.cuda.is_available()` before image training.
- macOS: `uv sync` selects ordinary PyTorch wheels; Apple Silicon may use MPS
  if the installed build exposes it, otherwise CPU is supported.
- Linux: the non-Windows dependency marker selects ordinary PyTorch wheels;
  choose an appropriate CUDA/ROCm environment if GPU training is needed.
- Serving and the dashboard do not require a GPU.
- MapLibre OSM tiles require network access; Radar remains available if tiles
  are unavailable.

## Credential handling

`.env`, `data/secrets/`, and `data/live/` are ignored. Use a short-lived
Earthdata token rather than a password. Rotate any password previously pasted
into chat. The application never sends a token to the browser; the NASA access
verifier follows one approved redirect without forwarding the bearer token to
the CDN.
