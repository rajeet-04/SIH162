"""Resumable historical FIRMS acquisition for research, without inferred labels."""

import argparse
import csv
import hashlib
import io
import json
import time
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

from thermis.live import normalize


def validate_keys(keys, http, interval=1.0):
    valid = []
    for slot, key in enumerate(dict.fromkeys(keys), 1):
        try:
            response = http.get(
                f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{key}/ALL"
            )
        except httpx.HTTPError:
            raise RuntimeError("FIRMS credential preflight transport error") from None
        if response.status_code == 429:
            raise RuntimeError("FIRMS rate limit reached during preflight; retry later")
        accepted = response.status_code == 200 and response.text.startswith("data_id,")
        if accepted:
            valid.append(key)
        print(json.dumps(dict(key_slot=slot, accepted=accepted)), flush=True)
        time.sleep(interval)
    if not valid:
        raise ValueError("No validated FIRMS keys available")
    return valid


class FirmsArchiveClient:
    """Rotate healthy keys at a conservative aggregate rate; stop on rate limiting."""

    def __init__(self, keys, http, interval=1.0):
        self.keys = list(dict.fromkeys(k.strip() for k in keys if k.strip()))
        if not self.keys:
            raise ValueError("No FIRMS credentials configured")
        self.http, self.interval = http, max(0, interval)
        self.counter, self.last, self.stopped = 0, 0.0, False

    def fetch(self, product, start, days):
        if self.stopped:
            raise RuntimeError("FIRMS acquisition stopped; retry later")
        if product not in ("VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "MODIS_SP") or not 1 <= days <= 5:
            raise ValueError("Unsupported research product or day range")
        time.sleep(max(0, self.last + self.interval - time.monotonic()))
        key = self.keys[self.counter % len(self.keys)]
        self.counter += 1
        self.last = time.monotonic()
        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/"
            f"{product}/68,4,98,38/{days}/{start.isoformat()}"
        )
        try:
            with self.http.stream("GET", url) as response:
                if response.status_code == 429:
                    self.stopped = True
                    raise RuntimeError("FIRMS rate limit reached; all key use paused")
                if response.status_code != 200:
                    raise RuntimeError(f"FIRMS HTTP {response.status_code}")
                raw = bytearray()
                for block in response.iter_bytes(65536):
                    raw.extend(block)
                    if len(raw) > 50_000_000:
                        raise ValueError("FIRMS response exceeds 50 MB bound")
        except httpx.HTTPError:
            raise RuntimeError("FIRMS transport error; credential URL suppressed") from None
        text = bytes(raw).decode("utf-8-sig")
        fields = csv.DictReader(io.StringIO(text)).fieldnames or []
        if not {"latitude", "longitude", "acq_date", "acq_time", "frp"}.issubset(fields):
            self.stopped = True
            raise ValueError("FIRMS schema rejected; acquisition stopped")
        return bytes(raw)


def download_archive(client, output, start, end, product="VIIRS_SNPP_SP"):
    if start > end or end >= date.today():
        raise ValueError("Only completed ascending historical dates are allowed")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = dict(product=product, start=str(start), end=str(end), bbox=[68, 4, 98, 38])
    spec_path = output / "request.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError("Output belongs to a different acquisition request")
    spec_path.write_text(json.dumps(spec, indent=2))
    day, chunks, records, reused = start, [], [], 0
    while day <= end:
        last = min(day + timedelta(days=4), end)
        path = output / f"{product}_{day}_{last}.csv"
        digest_path = path.with_suffix(".sha256")
        cached = path.exists()
        if cached:
            raw = path.read_bytes()
            if (
                not digest_path.exists()
                or hashlib.sha256(raw).hexdigest() != digest_path.read_text()
            ):
                raise ValueError("Cached FIRMS checksum mismatch; preserve and inspect")
            reused += 1
        else:
            raw = client.fetch(product, day, (last - day).days + 1)
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        batch = []
        for row in rows:
            if not str(day) <= row["acq_date"] <= str(last):
                raise ValueError("Observation outside requested dates")
            event = normalize(row, product)
            event["firms_type"] = row.get("type")  # provider flag, NOT reference class
            event["source_chunk"] = path.name
            batch.append(event)
        digest = hashlib.sha256(raw).hexdigest()
        if not path.exists():
            path.write_bytes(raw)
            digest_path.write_text(digest)
        records.extend(batch)
        chunks.append(dict(file=path.name, sha256=digest, rows=len(batch)))
        print(json.dumps(dict(start=str(day), rows=len(batch), cached=cached)), flush=True)
        day = last + timedelta(days=1)
    frame = pd.DataFrame(records)
    before = len(frame)
    if before:
        frame = frame.drop_duplicates("event_id").sort_values(["timestamp_utc", "event_id"])
    frame.to_parquet(output / "observations.parquet", index=False)
    report = dict(
        **spec,
        rows=len(frame),
        duplicates=before - len(frame),
        chunks=chunks,
        reused_chunks=reused,
        label_status="unlabeled; FIRMS type is not ground truth",
        acquired_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
    )
    (output / "manifest.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys-file", type=Path, default=Path("D:/data/firms_india/.map_keys"))
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--product", default="VIIRS_SNPP_SP")
    args = parser.parse_args()
    keys = [
        k for k in args.keys_file.read_text().splitlines() if k.strip() and not k.startswith("#")
    ]
    with httpx.Client(timeout=60, follow_redirects=False) as http:
        keys = validate_keys(keys, http)
        report = download_archive(
            FirmsArchiveClient(keys, http), args.output, args.start, args.end, args.product
        )
    print(json.dumps({k: v for k, v in report.items() if k != "chunks"}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from None
