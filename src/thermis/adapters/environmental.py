import re
from datetime import date
from pathlib import Path


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
