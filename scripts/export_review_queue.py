"""Export the expert review queue that unblocks ranking promotion.

Includes every sealed ranking row plus a deterministic sample of uncertain
development rows, with full context and empty expert-label columns.
Experts fill expert_label_stage_1/2; rerun evaluate-ranking afterwards.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "review_queue.csv"
SAMPLE_UNCERTAIN = 500
SEED = 26162

EXPERT_COLUMNS = [
    "event_id", "split", "timestamp_utc", "latitude", "longitude",
    "source_dataset", "frp", "brightness_temperature", "frp_uncertainty",
    "Day_night", "n_cloud", "n_water",
    "prior_detections_7d", "prior_detections_30d", "prior_detections_90d",
    "stationary_count_90d", "nearest_flare_distance_m",
    "nearest_industrial_distance_m", "label_stage_1", "label_stage_2",
    "label_source", "label_confidence", "label_provenance",
    "expert_label_stage_1", "expert_label_stage_2", "expert_notes",
]


def main() -> None:
    labels = pd.read_parquet(ROOT / "data/labels/events_labeled.parquet")
    splits = pd.read_parquet(ROOT / "data/splits/event_splits.parquet")[
        ["event_id", "split"]
    ]
    frame = labels.merge(splits, on="event_id", validate="one_to_one")
    ranking = frame[frame["split"] == "ranking"]
    uncertain_dev = frame[
        (frame["split"] == "development")
        & (frame["label_stage_2"] == "uncertain_review_required")
    ].sample(n=min(SAMPLE_UNCERTAIN, len(frame)), random_state=SEED)
    queue = pd.concat([ranking, uncertain_dev], ignore_index=True)
    queue["expert_label_stage_1"] = ""
    queue["expert_label_stage_2"] = ""
    queue["expert_notes"] = ""
    queue[EXPERT_COLUMNS].to_csv(OUT, index=False)
    print(f"ranking={len(ranking)} uncertain_sample={len(uncertain_dev)} -> {OUT}")


if __name__ == "__main__":
    main()
