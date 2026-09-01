from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


@dataclass(frozen=True)
class FireAtlasArchiveMetadata:
    path: str
    year: int
    kind: str
    record_count: int
    fields: tuple[str, ...]
    checksum_ok: bool


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


def verify_sha256_sidecar(path: Path) -> bool:
    sidecar = Path(f"{path}.sha256")
    if not sidecar.exists():
        return False
    expected = sidecar.read_text(encoding="ascii").strip().split()[0].lower()
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def inspect_fire_atlas_zip(path: Path) -> FireAtlasArchiveMetadata:
    """Inspect DBF metadata inside an archive without extracting it to disk."""
    import shapefile

    name = path.name.lower()
    kind = "ignition" if "ignition" in name else "perimeter"
    year_match = next(
        (part for part in path.stem.split("_") if part.isdigit() and len(part) == 4),
        None,
    )
    if year_match is None:
        raise ValueError(f"cannot parse year from {path.name}")
    with ZipFile(path) as archive:
        members = {member.lower(): member for member in archive.namelist()}
        dbf_name = next((name for name in members if name.endswith(".dbf")), None)
        if dbf_name is None:
            raise ValueError(f"archive has no DBF: {path.name}")
        reader = shapefile.Reader(dbf=BytesIO(archive.read(members[dbf_name])))
        fields = tuple(field[0] for field in reader.fields[1:])
        record_count = len(reader)
    return FireAtlasArchiveMetadata(
        path=str(path),
        year=int(year_match),
        kind=kind,
        record_count=record_count,
        fields=fields,
        checksum_ok=verify_sha256_sidecar(path),
    )


def build_fire_atlas_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*.zip")):
        rows.append(inspect_fire_atlas_zip(path).__dict__)
    return pd.DataFrame(rows)
