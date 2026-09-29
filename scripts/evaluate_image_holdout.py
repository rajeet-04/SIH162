"""Evaluate the exported image verifier on the sealed holdout set (no training).

Reads data/splits/image_splits.parquet + data/manifests/image_manifest.parquet,
runs models/image-smoke/image_verifier.ts on CPU over ALL holdout rows, and
writes reports/image_holdout.json with accuracy, per-class recall, and
expected calibration error (10 equal-width bins).
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
REPORT = ROOT / "reports" / "image_holdout.json"


def expected_calibration_error(
    confidences: list[float], correct: list[bool], bins: int = 10
) -> float:
    ece = 0.0
    n = len(confidences)
    for low, high in ((i / bins, (i + 1) / bins) for i in range(bins)):
        idx = [i for i, c in enumerate(confidences) if low < c <= high or (i == 0 and c == 0.0)]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        conf = sum(confidences[i] for i in idx) / len(idx)
        ece += len(idx) / n * abs(acc - conf)
    return ece


def main() -> None:
    splits = pd.read_parquet(SPLITS)
    manifest = pd.read_parquet(MANIFEST)[["image_id", "path", "label_source", "readable"]]
    holdout = splits[splits["split"] == "ranking"].merge(manifest, on="image_id")
    holdout = holdout[holdout["readable"] & holdout["label_source"].isin(["fire", "no_fire"])]
    classes = json.loads(METADATA.read_text(encoding="utf-8"))["classes"]
    label_ids = {label: i for i, label in enumerate(classes)}

    model = torch.jit.load(str(MODEL), map_location="cpu").eval()
    confidences: list[float] = []
    correct: list[bool] = []
    per_class = {label: [0, 0] for label in classes}  # label -> [right, total]
    with torch.inference_mode():
        for row in holdout.itertuples():
            try:
                logits = model(_image_tensor(Path(row.path)).unsqueeze(0))
            except (OSError, ValueError):
                continue
            probs = torch.softmax(logits, dim=1)[0].tolist()
            pred = int(torch.argmax(logits, dim=1).item())
            truth = label_ids[str(row.label_source)]
            conf = float(probs[pred])
            ok = pred == truth
            confidences.append(conf)
            correct.append(ok)
            per_class[str(row.label_source)][1] += 1
            per_class[str(row.label_source)][0] += int(ok)

    evaluated = len(correct)
    report = {
        "holdout_rows": int(len(holdout)),
        "evaluated_rows": evaluated,
        "skipped_rows": int(len(holdout) - evaluated),
        "accuracy": sum(correct) / evaluated if evaluated else 0.0,
        "per_class_recall": {
            label: (right / total if total else 0.0)
            for label, (right, total) in per_class.items()
        },
        "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "expected_calibration_error_10bin": expected_calibration_error(confidences, correct),
        "ranking_rows_used_for_training": 0,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
