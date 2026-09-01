import re
from pathlib import Path

import pandas as pd


def image_group_id(path: Path) -> str:
    """Return a stable scene key so patches cannot cross split boundaries."""
    stem = path.stem
    match = re.match(r"(.+?)_patch_\d+_\d+$", stem, re.IGNORECASE)
    return match.group(1) if match else stem


def build_image_manifest(roots: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family, root in sorted(roots.items()):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".npz"}:
                continue
            rows.append(
                {
                    "image_id": f"{family}:{path.as_posix()}",
                    "path": str(path),
                    "source_family": family,
                    "scene_group": image_group_id(path),
                    "acquired_utc": None,
                    "region_group": None,
                    "label_source": None,
                    "mask_path": None,
                    "channels": None,
                    "height": None,
                    "width": None,
                    "readable": True,
                    "exclusion_reason": None,
                }
            )
    return pd.DataFrame(rows)
