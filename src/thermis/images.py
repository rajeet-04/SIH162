import re
from pathlib import Path


def image_group_id(path: Path) -> str:
    """Return a stable scene key so patches cannot cross split boundaries."""
    stem = path.stem
    match = re.match(r"(.+?)_patch_\d+_\d+$", stem, re.IGNORECASE)
    return match.group(1) if match else stem
