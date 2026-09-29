from thermis.adapters.fire_atlas import FireAtlasArchive, assert_fire_atlas_join


def test_fire_atlas_ignition_and_perimeter_are_joined_not_stacked() -> None:
    ignition = FireAtlasArchive(year=2003, kind="ignition", record_count=989832)
    perimeter = FireAtlasArchive(year=2003, kind="perimeter", record_count=989861)
    report = assert_fire_atlas_join(ignition, perimeter)
    assert report.left_records == 989832
    assert report.right_records == 989861
    assert report.maximum_output_records == 989861
