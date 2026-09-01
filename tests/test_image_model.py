import pandas as pd

from thermis.image_model import ImageRecord, assert_scene_isolation, assign_image_splits


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
