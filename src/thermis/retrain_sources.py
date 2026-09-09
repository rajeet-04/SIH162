"""Bounded public-source acquisition; no secrets are embedded in artifact metadata."""

import hashlib
import json
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pandas as pd


def osm_points(payload: dict, as_of: str) -> pd.DataFrame:
    records = []
    for element in payload.get("elements", []):
        location = element.get("center", element)
        if "lat" not in location or "lon" not in location:
            continue
        tags = element.get("tags", {})
        kind = "flare" if tags.get("man_made") == "flare" else "industrial"
        records.append(
            dict(
                latitude=location["lat"],
                longitude=location["lon"],
                available_from=as_of,
                kind=kind,
                source_id=f"osm:{element['type']}/{element['id']}",
                tags=json.dumps(tags, sort_keys=True),
            )
        )
    return pd.DataFrame(
        records, columns=["latitude", "longitude", "available_from", "kind", "source_id", "tags"]
    )


def fetch_historical_facilities(
    output: Path, as_of="2026-06-01T00:00:00Z", reuse_from: Path | None = None
) -> dict:
    """Six bounded industrial study areas, explicitly not nationwide completeness."""
    if reuse_from is not None:
        previous = json.loads((reuse_from / "report.json").read_text(encoding="utf-8"))
        if pd.Timestamp(previous["as_of"]) != pd.Timestamp(as_of):
            raise ValueError("Cannot reuse a different historical snapshot")
    output.mkdir(parents=True, exist_ok=False)
    areas = {
        "jamnagar": (21.7, 69.3, 22.8, 70.8),
        "mumbai": (18.5, 72.5, 19.5, 73.5),
        "surat": (20.7, 72.3, 21.7, 73.5),
        "raipur": (20.7, 80.9, 22.0, 82.2),
        "bokaro": (23.1, 85.5, 24.1, 87.0),
        "paradip": (19.7, 86.0, 20.9, 87.2),
    }
    frames, status = [], {}
    with httpx.Client(timeout=60, headers={"User-Agent": "THERMIS-research/0.3"}) as client:
        for name, bounds in areas.items():
            bbox = ",".join(map(str, bounds))
            query = (
                f'[out:json][timeout:45][date:"{as_of}"];('
                f'nwr["industrial"]({bbox});nwr["landuse"="industrial"]({bbox});'
                f'nwr["man_made"="flare"]({bbox}););out center;'
            )
            try:
                cached = reuse_from / f"{name}.json" if reuse_from else None
                reused = False
                payload = None
                if cached is not None and cached.is_file():
                    try:
                        candidate = json.loads(cached.read_text(encoding="utf-8"))
                        if isinstance(candidate.get("elements"), list) and not candidate.get(
                            "remark"
                        ):
                            payload, reused = candidate, True
                    except (ValueError, OSError):
                        pass
                if payload is None:
                    response = client.post(
                        "https://overpass-api.de/api/interpreter", data={"data": query}
                    )
                    response.raise_for_status()
                    payload = response.json()
                if payload.get("remark"):
                    raise ValueError("Overpass reported incomplete query")
                if not isinstance(payload.get("elements"), list):
                    raise ValueError("Overpass response lacks elements")
                (output / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
                frame = osm_points(payload, as_of)
                frame["study_area"] = name
                frames.append(frame)
                status[name] = {
                    "status": "ok",
                    "features": len(frame),
                    "bbox": bounds,
                    "reused": reused,
                }
            except (httpx.HTTPError, ValueError) as error:
                status[name] = {"status": "failed", "error": type(error).__name__, "bbox": bounds}
            print(f"OSM {name}: {status[name]['status']}", flush=True)
    frame = pd.concat(frames, ignore_index=True) if frames else osm_points({}, as_of)
    frame = frame.drop_duplicates("source_id")
    frame.to_parquet(output / "facilities.parquet", index=False)
    report = {
        "as_of": as_of,
        "areas": status,
        "features": len(frame),
        "reuse_from": str(reuse_from) if reuse_from else None,
        "coverage": "six study areas only; OSM incompleteness is not negative evidence",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def recover_archive(
    url: str, destination: Path, expected_md5: str, maximum_bytes: int = 11_000_000_000
) -> dict:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "zenodo.org":
        raise ValueError("Untrusted recovery origin")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    digest, total = hashlib.md5(), 0
    # Exclusive creation preserves unsuccessful earlier attempts for inspection.
    with partial.open("xb") as handle, httpx.Client(timeout=60) as client:
        with client.stream("GET", url, follow_redirects=False) as response:
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError("Recovery did not return a complete file")
            for chunk in response.iter_bytes(1024 * 1024):
                total += len(chunk)
                if total > maximum_bytes:
                    raise ValueError("Archive exceeds download budget")
                digest.update(chunk)
                handle.write(chunk)
    if digest.hexdigest() != expected_md5:
        raise ValueError("Provider checksum mismatch; partial retained, not admitted")
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC failure")
        count = len(archive.namelist())
    partial.rename(destination)
    return {
        "path": str(destination),
        "bytes": total,
        "md5": expected_md5,
        "members": count,
        "crc_verified": True,
    }
