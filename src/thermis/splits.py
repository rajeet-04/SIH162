import math

import pandas as pd


def make_grouped_time_split(
    frame: pd.DataFrame,
    time_col: str,
    group_cols: list[str],
    ranking_fraction: float = 0.10,
) -> pd.DataFrame:
    if not 0 < ranking_fraction < 1:
        raise ValueError("ranking_fraction must be between zero and one")
    result = frame.copy()
    result[time_col] = pd.to_datetime(result[time_col], utc=True)
    group_key = result[group_cols].astype(str).agg("|".join, axis=1)
    result["split_group"] = group_key
    summary = result.groupby("split_group", sort=False)[time_col].agg(
        first_timestamp="min", last_timestamp="max", row_count="size"
    )
    summary = summary.sort_values("last_timestamp")
    target_rows = max(1, math.ceil(len(result) * ranking_fraction))
    ranking_groups: list[str] = []
    ranking_rows = 0
    for group, row in reversed(list(summary.iterrows())):
        ranking_groups.append(group)
        ranking_rows += int(row["row_count"])
        if ranking_rows >= target_rows:
            break
    ranking_set = set(ranking_groups)
    result["split"] = result["split_group"].map(
        lambda group: "ranking" if group in ranking_set else "development"
    )
    result["split_reason"] = result["split"].map(
        {"ranking": "newest_whole_groups_reserved", "development": "older_90_percent"}
    )
    result = result.merge(summary, left_on="split_group", right_index=True, how="left")
    return result.sort_values(time_col).reset_index(drop=True)


def assert_no_group_overlap(split_manifest: pd.DataFrame, group_cols: list[str]) -> None:
    groups = split_manifest[group_cols].astype(str).agg("|".join, axis=1)
    overlap = split_manifest.assign(_group=groups).groupby("_group")["split"].nunique()
    if int((overlap > 1).sum()) > 0:
        raise ValueError("split group appears in more than one split")


def assert_ranking_is_newest(split_manifest: pd.DataFrame, time_col: str) -> None:
    timestamps = pd.to_datetime(split_manifest[time_col], utc=True)
    ranking = timestamps[split_manifest["split"] == "ranking"]
    development = timestamps[split_manifest["split"] != "ranking"]
    # Satellite products commonly timestamp several detections at the same
    # instant. Equal boundary timestamps are valid; ranking may not be older.
    if ranking.empty or development.empty or ranking.min() < development.max():
        raise ValueError("ranking split contains an event older than development")
