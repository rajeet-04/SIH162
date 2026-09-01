from dataclasses import dataclass


@dataclass(frozen=True)
class FireAtlasArchive:
    year: int
    kind: str
    record_count: int


@dataclass(frozen=True)
class FireAtlasJoinReport:
    left_records: int
    right_records: int
    maximum_output_records: int


def assert_fire_atlas_join(
    ignition: FireAtlasArchive, perimeter: FireAtlasArchive
) -> FireAtlasJoinReport:
    if ignition.year != perimeter.year:
        raise ValueError("ignition and perimeter years must match")
    if ignition.kind != "ignition" or perimeter.kind != "perimeter":
        raise ValueError("archives must be ignition/perimeter pairs")
    return FireAtlasJoinReport(
        left_records=ignition.record_count,
        right_records=perimeter.record_count,
        maximum_output_records=max(ignition.record_count, perimeter.record_count),
    )
