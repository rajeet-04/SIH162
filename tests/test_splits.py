import pandas as pd

from thermis.splits import assert_no_group_overlap, make_grouped_time_split


def test_newest_ten_percent_is_ranking_and_groups_do_not_cross() -> None:
    frame = pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(20)],
            "timestamp_utc": pd.date_range("2024-01-01", periods=20, tz="UTC"),
            "scene_group": [f"s{i // 2}" for i in range(20)],
        }
    )
    split = make_grouped_time_split(frame, "timestamp_utc", ["scene_group"], 0.10)
    assert set(split.tail(2)["split"]) == {"ranking"}
    assert split.groupby("scene_group")["split"].nunique().max() == 1
    assert_no_group_overlap(split, ["scene_group"])
