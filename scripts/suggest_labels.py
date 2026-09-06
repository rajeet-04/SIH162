"""Pre-fill the expert review queue with tier-2 suggested labels.

Tier-2 rules read ONLY attributes outside the model feature set (FIRMS
confidence, day/night, brightness) plus basic physics, so suggestions are not
circular with the classifier. They never override tier-1 rule labels and never
become training labels by themselves: an expert confirms them into
expert_label_stage_1/2, which stays the sole authoritative source.

Reads data/labels/events_labeled.parquet + data/splits/event_splits.parquet,
writes reports/review_queue_suggested.csv (the 1,217 ranking rows plus the 500
uncertain dev sample, each with suggestion columns).
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "review_queue_suggested.csv"
SEED = 26162
SAMPLE_UNCERTAIN = 500


def safe_str(row: pd.Series, key: str) -> str:
    value = row.get(key)
    if value is None:
        return ""
    if not isinstance(value, str) and bool(pd.isna(value)):
        return ""
    return str(value)


def safe_float(row: pd.Series, key: str, default: float) -> float:
    try:
        value = float(row.get(key))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return value if value == value else default  # NaN guard


def suggest(row: pd.Series) -> tuple[str, str, str, float, str]:
    """Return (stage1, stage2, source, confidence, rationale) or blanks."""
    blank = ("", "", "", 0.0, "")
    if row.get("label_stage_2") != "uncertain_review_required":
        return blank
    conf = safe_str(row, "firms_confidence").strip().lower()
    daynight = safe_str(row, "firms_daynight").strip().upper()
    if not conf:
        return blank  # old rows lack independent sensor attributes
    flare = safe_float(row, "nearest_flare_distance_m", float("inf"))
    p90 = int(safe_float(row, "prior_detections_90d", 0))
    frp = safe_float(row, "frp", 0.0)
    night_flare = (
        daynight == "N" and flare <= 1000.0 and p90 >= 10,
        ("persistent_or_non_emergency_heat", "persistent_industrial_heat_or_flare",
         "night_flare_and_persistence", 0.80, f"N+flare{flare:.0f}m+p90={p90}"),
    )
    if night_flare[0]:
        return night_flare[1]
    if conf in {"h", "n"} and daynight == "D" and flare > 5000.0 and p90 <= 2 and frp >= 3.0:
        return ("emergency_or_transient_fire", "wildfire_or_agricultural_burn",
                "day_fire_far_from_flare", 0.65,
                f"conf={conf}+D+flare{flare:.0f}m+p90={p90}+frp={frp:.1f}")
    if conf == "l" and p90 == 0 and daynight == "D":
        return ("uncertain", "other_or_uncertain",
                "low_confidence_isolated_day", 0.60, "conf=l+p90=0+D")
    return blank


def main() -> None:
    labels = pd.read_parquet(ROOT / "data/labels/events_labeled.parquet")
    splits = pd.read_parquet(ROOT / "data/splits/event_splits.parquet")[["event_id", "split"]]
    frame = labels.merge(splits, on="event_id", validate="one_to_one")
    ranking = frame[frame["split"] == "ranking"]
    uncertain_dev = frame[
        (frame["split"] == "development")
        & (frame["label_stage_2"] == "uncertain_review_required")
    ].sample(n=min(SAMPLE_UNCERTAIN, len(frame)), random_state=SEED)
    queue = pd.concat([ranking, uncertain_dev], ignore_index=True)
    suggestions = [suggest(row) for _, row in queue.iterrows()]
    queue["suggested_stage_1"] = [s[0] for s in suggestions]
    queue["suggested_stage_2"] = [s[1] for s in suggestions]
    queue["suggestion_source"] = [s[2] for s in suggestions]
    queue["suggestion_confidence"] = [s[3] for s in suggestions]
    queue["suggestion_rationale"] = [s[4] for s in suggestions]
    queue["expert_label_stage_1"] = ""
    queue["expert_label_stage_2"] = ""
    queue["expert_notes"] = ""
    queue.to_csv(OUT, index=False)
    filled = queue[queue["suggested_stage_2"] != ""]
    print(f"queue={len(queue)} suggested={len(filled)} -> {OUT}")
    print("by split:")
    print(filled.groupby(queue["split"])["suggested_stage_2"].value_counts().to_string())
    print("suggestion_confidence mean:", round(float(filled["suggestion_confidence"].mean()), 3))


if __name__ == "__main__":
    main()
