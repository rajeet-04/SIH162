from thermis.image_model import ImageRecord, assert_scene_isolation


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
