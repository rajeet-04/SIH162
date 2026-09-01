import re
from datetime import date
from pathlib import Path

import pandas as pd


def parse_micasa_date(path: Path) -> date:
    match = re.search(r"_(\d{8})\.nc4$", path.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot find YYYYMMDD in {path.name}")
    value = match.group(1)
    return date(int(value[:4]), int(value[4:6]), int(value[6:]))


def parse_modis_identity(path: Path) -> tuple[int, int, str, str]:
    match = re.search(r"\.A(\d{4})(\d{3})\.(h\d{2}v\d{2})\.(\d{3})\.", path.name)
    if not match:
        raise ValueError(f"cannot parse MODIS identity from {path.name}")
    return int(match.group(1)), int(match.group(2)), match.group(3), match.group(4)


def build_environment_manifest(roots: dict[str, Path]) -> pd.DataFrame:
    """Create metadata rows from filenames; data arrays are never loaded."""
    rows: list[dict[str, object]] = []
    for family, root in sorted(roots.items()):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            row: dict[str, object] = {
                "path": str(path),
                "source_family": family,
                "modified_utc": pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC"),
                "metadata_status": "unclassified",
                "acquisition_day": None,
                "tile": None,
                "collection": None,
            }
            try:
                if path.suffix.lower() == ".nc4":
                    row["acquisition_day"] = parse_micasa_date(path).isoformat()
                    row["metadata_status"] = "filename_parsed"
                elif path.suffix.lower() == ".hdf":
                    year, doy, tile, collection = parse_modis_identity(path)
                    row.update(
                        acquisition_day=f"{year}-{doy:03d}", tile=tile, collection=collection
                    )
                    row["metadata_status"] = "filename_parsed"
            except ValueError:
                row["metadata_status"] = "unparsed"
            rows.append(row)
    return pd.DataFrame(rows)
