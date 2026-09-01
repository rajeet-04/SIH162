from hashlib import blake2b
from pathlib import Path

import pandas as pd

_INCOMPLETE_SUFFIXES = {".part", ".tmp", ".crdownload"}


def fast_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash file edges and size for a fast, deterministic duplicate signal."""
    digest = blake2b(digest_size=16)
    with path.open("rb") as handle:
        digest.update(handle.read(chunk_size))
        if path.stat().st_size > chunk_size:
            handle.seek(max(0, path.stat().st_size - chunk_size))
            digest.update(handle.read(chunk_size))
    digest.update(str(path.stat().st_size).encode("ascii"))
    return digest.hexdigest()


def inventory_root(root: Path, source_family: str) -> pd.DataFrame:
    """Inventory files without changing the source root."""
    rows: list[dict[str, object]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        excluded = path.suffix.lower() in _INCOMPLETE_SUFFIXES
        rows.append(
            {
                "path": str(path),
                "source_family": source_family,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "modified_utc": pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC"),
                "fast_hash": fast_hash(path),
                "status": "excluded" if excluded else "candidate",
                "exclusion_reason": "incomplete_download" if excluded else None,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "path",
            "source_family",
            "extension",
            "size_bytes",
            "modified_utc",
            "fast_hash",
            "status",
            "exclusion_reason",
        ],
    )


def inventory_sources(settings: object) -> pd.DataFrame:
    """Inventory every configured source root in stable source-key order."""
    frames = [
        inventory_root(root, source_family=name)
        for name, root in sorted(settings.sources.items())
        if root.exists()
    ]
    if not frames:
        return inventory_root(Path("."), source_family="empty").iloc[0:0]
    return pd.concat(frames, ignore_index=True)
