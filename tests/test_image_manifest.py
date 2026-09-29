from pathlib import Path

from thermis.images import image_group_id


def test_npz_patches_share_scene_group() -> None:
    assert image_group_id(Path("scene_1_patch_1_10.npz")) == "scene_1"
    assert image_group_id(Path("scene_1_patch_7_3.npz")) == "scene_1"
