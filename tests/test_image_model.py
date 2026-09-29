import pandas as pd

from thermis.image_model import (
    ImageRecord,
    _grouped_dev_val_split,
    assert_scene_isolation,
    assign_image_splits,
)


def test_scene_isolation_rejects_cross_split_scene() -> None:
    records = [
        ImageRecord("a", "scene-1", "train", "fire", "a.jpg"),
        ImageRecord("b", "scene-1", "ranking", "fire", "b.jpg"),
    ]
    try:
        assert_scene_isolation(records)
    except ValueError as exc:
        assert "scene-1" in str(exc)
    else:
        raise AssertionError("expected scene overlap rejection")


def test_image_quality_rejects_missing_path(tmp_path) -> None:
    from thermis.image_model import inspect_image

    result = inspect_image(tmp_path / "missing.jpg")
    assert result.readable is False


def test_image_split_keeps_scene_groups_together() -> None:
    frame = pd.DataFrame(
        {
            "image_id": ["a", "b", "c", "d"],
            "scene_group": ["scene-a", "scene-a", "scene-b", "scene-c"],
        }
    )
    split = assign_image_splits(frame, ranking_fraction=0.34)
    assert split.groupby("scene_group")["split"].nunique().max() == 1
    assert set(split["split"]) == {"development", "ranking"}


def test_grouped_dev_val_split_isolates_newest_groups() -> None:
    records = [
        ImageRecord(f"id{i}", f"scene-{g}", "development", "fire", f"{i}.jpg")
        for g in range(4)
        for i in range(g * 2, g * 2 + 2)
    ]
    fit, val = _grouped_dev_val_split(records, 0.25)
    fit_groups = {record.scene_group for record in fit}
    val_groups = {record.scene_group for record in val}
    assert fit_groups.isdisjoint(val_groups)
    assert val_groups == {"scene-3"}
    assert len(fit) == 6 and len(val) == 2
