import json
from pathlib import Path

import pandas as pd


def build_demo_bundle(
    labels_path: Path = Path("data/labels/events_labeled.parquet"),
    output_path: Path = Path("data/demo/events.json"),
) -> None:
    frame = pd.read_parquet(labels_path)
    selected = []
    for label, demo_id in (
        ("industrial_fire_candidate", "demo-industrial-fire"),
        ("persistent_industrial_heat_or_flare", "demo-persistent-heat"),
        ("uncertain_review_required", "demo-uncertain"),
    ):
        match = frame[frame["label_stage_2"] == label]
        if match.empty:
            continue
        india = match[match["latitude"].between(6, 38) & match["longitude"].between(66, 100)]
        row = (india if not india.empty else match).iloc[0]
        selected.append(
            {
                "event_id": demo_id,
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "prediction": {
                    "final_class": str(row.label_stage_2),
                    "confidence": float(row.label_confidence),
                    "model_version": "tabular-local-0.1-demo",
                },
                "evidence": {
                    "frp": float(row.frp),
                    "prior_detections_90d": int(row.prior_detections_90d),
                    "nearest_flare_distance_m": float(row.nearest_flare_distance_m),
                    "label_source": str(row.label_source),
                },
                "timeline_90d": [
                    {"days_ago": 90, "prior_detections": 0},
                    {"days_ago": 30, "prior_detections": int(row.prior_detections_30d)},
                    {"days_ago": 7, "prior_detections": int(row.prior_detections_7d)},
                    {"days_ago": 0, "frp": float(row.frp)},
                ],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({item["event_id"]: item for item in selected}, indent=2))


if __name__ == "__main__":
    build_demo_bundle()
