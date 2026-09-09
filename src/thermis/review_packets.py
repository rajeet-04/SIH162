"""Blind, exclusively owned development-review packets; not a test benchmark."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

EDITABLE = [
    "review_status",
    "label",
    "evidence_url",
    "evidence_date",
    "evidence_kind",
    "confidence",
    "notes",
    "reviewed_by",
    "reviewed_at_utc",
]
LABELS = {
    "persistent_industrial_heat_or_flare",
    "industrial_fire_candidate",
    "vegetation_fire_candidate",
    "agricultural_burn_candidate",
    "uncertain",
}


def build_packets(frame, output, reviewers, count=600, cutoff="2025-05-18T08:12:00Z"):
    if (
        not reviewers
        or len(set(reviewers)) != len(reviewers)
        or any(not re.fullmatch(r"[A-Za-z0-9-]+", r) for r in reviewers)
        or count < len(reviewers)
        or count % len(reviewers)
    ):
        raise ValueError("Unique GitHub usernames and equal positive assignment counts required")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite reviewer edits")
    f = frame.copy()
    f["timestamp_utc"] = pd.to_datetime(f.timestamp_utc, utc=True, errors="coerce")
    boundary = pd.to_datetime(cutoff, utc=True)
    for col in ("latitude", "longitude", "frp"):
        f[col] = pd.to_numeric(f[col], errors="coerce")
    f = f[
        f.timestamp_utc.lt(boundary)
        & f.event_id.notna()
        & f.latitude.between(4, 38)
        & f.longitude.between(68, 98)
        & f.frp.ge(0)
        & np.isfinite(f.frp)
    ].copy()
    f = f.sort_values(["timestamp_utc", "event_id"]).drop_duplicates("event_id")
    # Cells are a sampling device, not inferred facilities or physical event IDs.
    f["cell_id"] = (
        np.floor(f.latitude / 0.05).astype(int).astype(str)
        + "_"
        + np.floor(f.longitude / 0.05).astype(int).astype(str)
    )
    f["month"] = f.timestamp_utc.dt.strftime("%Y-%m")
    cells = sorted(f.cell_id.unique())
    if len(cells) < count:
        raise ValueError("Insufficient distinct cells")
    rng = np.random.default_rng(26162)
    chosen = rng.choice(cells, count, replace=False)
    groups = {key: group for key, group in f[f.cell_id.isin(chosen)].groupby("cell_id")}
    rows = []
    for i, cell in enumerate(chosen):
        group = groups[cell]
        month = rng.choice(sorted(group.month.unique()))
        window = group[group.month.eq(month)]
        rep = window.iloc[len(window) // 2]
        lat, lon = float(rep.latitude), float(rep.longitude)
        day = rep.timestamp_utc.strftime("%Y-%m-%d")
        rows.append(
            dict(
                review_id=f"RV1-{i + 1:04d}",
                owner=reviewers[i % len(reviewers)],
                purpose="development_only",
                cell_id=cell,
                window_month=month,
                representative_event_id=str(rep.event_id),
                representative_timestamp_utc=rep.timestamp_utc.isoformat(),
                latitude=lat,
                longitude=lon,
                first_detection_utc=window.timestamp_utc.min().isoformat(),
                last_detection_utc=window.timestamp_utc.max().isoformat(),
                observation_count=len(window),
                active_dates=window.timestamp_utc.dt.date.nunique(),
                frp_median_mw=float(window.frp.median()),
                frp_max_mw=float(window.frp.max()),
                worldview_url=(
                    f"https://worldview.earthdata.nasa.gov/?v={lon - 0.08:.5f},"
                    f"{lat - 0.08:.5f},{lon + 0.08:.5f},{lat + 0.08:.5f}&t={day}"
                ),
                osm_url=f"https://www.openstreetmap.org/?mlat={lat:.5f}&mlon={lon:.5f}#map=14/{lat:.5f}/{lon:.5f}",
                source_product="VIIRS_SNPP_SP",
                source_archive=f"firms-snpp-{rep.timestamp_utc.year}-v1",
            )
        )
    reference = pd.DataFrame(rows)
    output.mkdir(parents=True)
    reference.to_csv(output / "reference_index.csv", index=False)
    for reviewer in reviewers:
        packet = reference[reference.owner.eq(reviewer)].copy()
        for col in EDITABLE:
            packet[col] = "pending" if col == "review_status" else ""
        packet.to_csv(output / f"{reviewer}.csv", index=False)
    manifest = dict(
        version=1,
        seed=26162,
        reviewers=reviewers,
        rows=count,
        cutoff_exclusive=boundary.isoformat(),
        candidate_cells=len(cells),
        eligible_observations=len(f),
        purpose="development_only",
        sampling=(
            "Uniform cells without replacement; uniform observed month per cell; "
            "median-time representative"
        ),
        cell_degrees=0.05,
        no_physical_site_isolation_claim=True,
        independent_second_review=False,
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.csv")},
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return validate_packets(output)


def validate_packets(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    index_path = output / "reference_index.csv"
    if (
        hashlib.sha256(index_path.read_bytes()).hexdigest()
        != manifest["files"]["reference_index.csv"]
    ):
        raise ValueError("Changed immutable reference index")
    reference = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    seen, reviewed = set(), 0
    for reviewer in manifest["reviewers"]:
        packet = pd.read_csv(output / f"{reviewer}.csv", dtype=str, keep_default_na=False)
        expected = reference[reference.owner.eq(reviewer)].set_index("review_id")
        if (
            packet.review_id.duplicated().any()
            or set(packet.review_id) != set(expected.index)
            or seen.intersection(packet.review_id)
        ):
            raise ValueError("Missing, duplicate or overlapping assignment")
        seen.update(packet.review_id)
        actual = packet.set_index("review_id").sort_index()
        if not actual[expected.columns].equals(expected.sort_index()):
            raise ValueError("Changed immutable evidence or owner")
        for row in packet.to_dict("records"):
            if row["review_status"] not in {"pending", "reviewed"}:
                raise ValueError("Invalid review_status")
            if row["review_status"] == "pending":
                continue
            if row["label"] not in LABELS:
                raise ValueError("Invalid label")
            if row["label"] == "uncertain":
                if not row["notes"].strip():
                    raise ValueError("Uncertain reviews require notes")
            elif (
                not row["evidence_url"].startswith(("https://", "http://"))
                or not row["evidence_date"]
                or not row["evidence_kind"]
                or row["confidence"] not in {"low", "medium", "high"}
                or row["reviewed_by"] != reviewer
                or not row["reviewed_at_utc"]
            ):
                raise ValueError(
                    "Reviewed label requires dated evidence, confidence and reviewer identity"
                )
            reviewed += 1
    return dict(
        rows=len(seen),
        reviewed=reviewed,
        pending=len(seen) - reviewed,
        overlap=0,
        purpose="development_only",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--observations", nargs="+")
    parser.add_argument("--reviewers", nargs="+")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        result = validate_packets(args.output)
    else:
        if not args.observations or not args.reviewers:
            parser.error("Generation requires observations and reviewers")
        frame = pd.concat([pd.read_parquet(p) for p in args.observations], ignore_index=True)
        result = build_packets(frame, args.output, args.reviewers)
        path = Path(args.output) / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["source_sha256"] = {
            Path(p).name + ":" + Path(p).parent.name: hashlib.sha256(
                Path(p).read_bytes()
            ).hexdigest()
            for p in args.observations
        }
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
