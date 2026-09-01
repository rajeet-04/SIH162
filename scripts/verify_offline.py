"""Verify the checked-in SIH26162 MVP without network access."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from fastapi.testclient import TestClient

from thermis.api import ModelRuntime, create_app
from thermis.artifacts import load_bundle

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    features = pd.read_parquet(ROOT / "data/features/event_features.parquet")
    splits = pd.read_parquet(ROOT / "data/splits/event_splits.parquet")
    if len(features) != 9186 or len(splits) != 9186:
        raise SystemExit("unexpected prepared event row count")
    split_counts = splits["split"].value_counts().to_dict()
    if split_counts.get("development") != 8267 or split_counts.get("ranking") != 919:
        raise SystemExit(f"unexpected split counts: {split_counts}")
    image_splits = pd.read_parquet(ROOT / "data/splits/image_splits.parquet")
    image_split_counts = image_splits["split"].value_counts().to_dict()
    if sum(image_split_counts.values()) != 47992 or image_split_counts.get("ranking", 0) <= 0:
        raise SystemExit(f"unexpected image split counts: {image_split_counts}")
    if image_splits.groupby("scene_group")["split"].nunique().max() != 1:
        raise SystemExit("image scene group crossed split boundary")

    bundles = {}
    for stage in ("stage1", "stage2"):
        bundle = load_bundle(ROOT / "models/tabular" / f"{stage}.joblib")
        if bundle.get("ranking_rows_used") != 0:
            raise SystemExit(f"{stage} bundle consumed ranking rows")
        bundles[stage] = {
            "fit_rows": bundle["training_rows"],
            "calibration_rows": bundle["validation_rows"],
            "metrics": bundle["metrics"],
            "model": type(bundle["model"]).__name__,
        }

    image_model = torch.jit.load(
        str(ROOT / "models/image-smoke/image_verifier.ts"), map_location="cpu"
    )
    image_metadata = json.loads(
        (ROOT / "models/image-smoke/metadata.json").read_text(encoding="utf-8")
    )
    if image_metadata.get("training_rows") != 35437 or image_metadata.get("ranking_rows") != 12555:
        raise SystemExit(f"unexpected image training metadata: {image_metadata}")
    image_output = image_model(torch.zeros(1, 3, 224, 224))
    if tuple(image_output.shape) != (1, 2):
        raise SystemExit(f"unexpected image verifier output shape: {tuple(image_output.shape)}")

    demo = json.loads((ROOT / "data/demo/events.json").read_text(encoding="utf-8"))
    image_manifest = pd.read_parquet(ROOT / "data/manifests/image_manifest.parquet")
    sample_image = image_manifest.loc[
        image_manifest["readable"] & image_manifest["label_source"].isin(["fire", "no_fire"]),
        "path",
    ].iloc[0]
    runtime = ModelRuntime.from_paths(ROOT / "models/tabular", ROOT / "data/demo/events.json")
    client = TestClient(create_app(runtime))
    health = client.get("/health")
    events = client.get("/events")
    metrics = client.get("/metrics")
    prediction = client.post(
        "/predict",
        json={
            "latitude": 24.48,
            "longitude": 48.18,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "frp": 12.0,
            "brightness_temperature": 320.0,
            "frp_uncertainty": 1.0,
        },
    )
    image_verification = client.post("/verify-image", json={"path": str(sample_image)})
    if (
        health.status_code != 200
        or events.status_code != 200
        or metrics.status_code != 200
        or prediction.status_code != 200
        or image_verification.status_code != 200
    ):
        raise SystemExit("offline API rehearsal failed")
    if len(events.json()["events"]) != len(demo):
        raise SystemExit("demo event count mismatch")
    if health.json().get("image_verifier_loaded") is not True:
        raise SystemExit("image verifier was not connected to the API runtime")

    report = {
        "offline": True,
        "event_rows": len(features),
        "split_counts": split_counts,
        "tabular_bundles": bundles,
        "image_verifier": {
            "device": "cpu",
            "output_shape": list(image_output.shape),
            "training_rows": image_metadata["training_rows"],
            "ranking_rows": image_metadata["ranking_rows"],
            "batch_size": image_metadata["batch_size"],
            "amp": image_metadata["amp"],
        },
        "image_split_counts": image_split_counts,
        "demo_events": len(demo),
        "api": {
            "health": health.status_code,
            "events": events.status_code,
            "metrics": metrics.status_code,
            "predict": prediction.status_code,
            "verify_image": image_verification.status_code,
        },
        "ranking_promotion": "blocked_pending_authoritative_labels",
    }
    destination = ROOT / "reports/offline_verification.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
