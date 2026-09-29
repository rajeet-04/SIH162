"""Fit single-parameter temperature scaling on development rows only.

Uses a deterministic 2,000-row development sample (never holdout), optimizes
temperature T on NLL with LBFGS, and writes reports/image_calibration.json.
"""

import json
from pathlib import Path

import pandas as pd
import torch

from thermis.image_model import _image_tensor

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits" / "image_splits.parquet"
MANIFEST = ROOT / "data" / "manifests" / "image_manifest.parquet"
MODEL = ROOT / "models" / "image-smoke" / "image_verifier.ts"
METADATA = ROOT / "models" / "image-smoke" / "metadata.json"
REPORT = ROOT / "reports" / "image_calibration.json"


def main(sample_size: int = 2000, seed: int = 26162) -> None:
    splits = pd.read_parquet(SPLITS)
    manifest = pd.read_parquet(MANIFEST)[["image_id", "path", "label_source", "readable"]]
    dev = splits[splits["split"] == "development"].merge(manifest, on="image_id")
    dev = dev[dev["readable"] & dev["label_source"].isin(["fire", "no_fire"])]
    dev = dev.sample(n=min(sample_size, len(dev)), random_state=seed).reset_index(drop=True)
    classes = json.loads(METADATA.read_text(encoding="utf-8"))["classes"]
    label_ids = {label: i for i, label in enumerate(classes)}

    model = torch.jit.load(str(MODEL), map_location="cpu").eval()
    logits_list, targets = [], []
    with torch.inference_mode():
        for count, row in enumerate(dev.itertuples()):
            try:
                logits_list.append(model(_image_tensor(Path(row.path)).unsqueeze(0))[0])
            except (OSError, ValueError):
                continue
            targets.append(label_ids[str(row.label_source)])
            if (count + 1) % 500 == 0:
                print(f"collected {count + 1}/{len(dev)} dev logits", flush=True)
    print(f"collected {len(targets)} dev logits, fitting temperature", flush=True)
    logits = torch.stack(logits_list)
    targets_t = torch.tensor(targets)

    def nll(temp: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(logits / temp, targets_t)

    with torch.inference_mode():
        nll_before = float(nll(torch.tensor(1.0)))
    temperature = torch.tensor(1.0, requires_grad=True)
    optimizer = torch.optim.LBFGS([temperature], lr=0.5, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = nll(temperature.clamp_min(0.05))
        loss.backward()
        return loss

    optimizer.step(closure)
    fitted = float(temperature.clamp_min(0.05).detach())
    with torch.inference_mode():
        nll_after = float(nll(torch.tensor(fitted)))

    payload = {
        "temperature": fitted,
        "fit_rows": len(targets),
        "fit_split": "development_sample_only",
        "nll_before": nll_before,
        "nll_after": nll_after,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
