"""Fetch NASA FIRMS thermal detections for the India window.

Needs a free MAP_KEY: register an email at
https://firms.modaps.eosdis.nasa.gov/api/area/ , then run::

    $env:FIRMS_MAP_KEY = "<key>"
    .\\.venv\\Scripts\\python.exe scripts/fetch_firms_india.py

Output CSVs land in D:/data/firms_india/ using the exact column layout the
FRP adapter already ingests (latitude, longitude, time, BT_MIR, FRP_MWIR,
FRP_uncertainty_MWIR, source_file), so prepare-events picks them up untouched.
India window: 66E-100E, 4N-39N (mainland plus island territories and margin).
"""

import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MAP_KEY = os.environ.get("FIRMS_MAP_KEY", "").strip()
KEYS_FILE = os.environ.get("FIRMS_KEYS_FILE", "").strip()
KEYS_ENV = os.environ.get("FIRMS_MAP_KEYS", "").strip()
DAY_RANGE = os.environ.get("FIRMS_DAY_RANGE", "5")
OUT_DIR = Path(os.environ.get("FIRMS_OUT_DIR", r"D:/data/firms_india"))
BBOX = "66,4,100,39"  # west,south,east,north
PRODUCTS = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "MODIS_NRT"]


def candidate_keys() -> list[str]:
    """Collect keys from file/env/single var, one per line, deduped.

    The file form exists so pasted keys never touch shell history or logs.
    Each line is one key; inner whitespace from copy-paste is stripped.
    Keys are used verbatim (any length) — the server accepts or rejects.
    """
    lines: list[str] = []
    if KEYS_FILE and Path(KEYS_FILE).exists():
        lines += Path(KEYS_FILE).read_text(encoding="utf-8").splitlines()
    lines += KEYS_ENV.replace(",", "\n").splitlines()
    lines.append(MAP_KEY)
    seen: list[str] = []
    for line in lines:
        key = line.replace(" ", "").replace("\t", "").strip().rstrip(";")
        if len(key) >= 16 and key not in seen:
            seen.append(key)
    return seen


def fetch_product(product: str, keys: list[str]) -> pd.DataFrame:
    print(f"GET {product}", flush=True)
    failures: list[str] = []
    for index in range(len(keys)):
        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            f"{keys[index]}/{product}/{BBOX}/{DAY_RANGE}"
        )
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                text = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:160].replace("\n", " ")
            failures.append(f"key{index + 1}:http{error.code}:{detail}")
            continue
        except Exception:  # network failure; try next key
            failures.append(f"key{index + 1}:transport")
            continue
        if text.startswith(("Invalid", "Error", "<")):
            failures.append(f"key{index + 1}:rejected")
            continue
        path = OUT_DIR / f"firms_{product.lower()}_india_raw.csv"
        path.write_text(text, encoding="utf-8")
        frame = pd.read_csv(path)
        print(f"  {len(frame)} rows via key{index + 1} -> {path.name}", flush=True)
        return frame
    raise SystemExit(f"FIRMS {product}: all keys exhausted ({';'.join(failures)})")


def normalize(raw: pd.DataFrame, product: str) -> pd.DataFrame:
    frame = raw.copy()
    frame["time"] = pd.to_datetime(
        frame["acq_date"].astype(str) + " " + frame["acq_time"].astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M",
        utc=True,
    )
    brightness = (
        frame["bright_ti4"] if "bright_ti4" in frame else frame.get("brightness")
    )
    def text(column: str) -> object:
        return frame[column].astype(str) if column in frame else None

    out = pd.DataFrame(
        {
            "latitude": frame["latitude"],
            "longitude": frame["longitude"],
            "time": frame["time"],
            "BT_MIR": brightness,
            "FRP_MWIR": frame.get("frp"),
            # FIRMS ships no per-pixel FRP uncertainty; 0.0 keeps the
            # ThermalEvent contract (frp_uncertainty >= 0) satisfied.
            "FRP_uncertainty_MWIR": 0.0,
            "source_file": f"firms_{product.lower()}_india.csv",
            "firms_confidence": text("confidence"),
            "firms_satellite": text("satellite"),
            "firms_daynight": text("daynight"),
        }
    )
    return out.dropna(subset=["latitude", "longitude", "time"])


def main() -> None:
    keys = candidate_keys()
    if not keys:
        raise SystemExit("Set FIRMS_MAP_KEY first (see module docstring).")
    print(f"keys_loaded={len(keys)}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for product in PRODUCTS:
        raw = fetch_product(product, keys)
        if raw.empty:
            continue
        clean = normalize(raw, product)
        dest = OUT_DIR / f"firms_{product.lower()}_india.csv"
        clean.to_csv(dest, index=False)
        total += len(clean)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (OUT_DIR / f"fetch_{stamp}.log").write_text(
        f"products={','.join(PRODUCTS)} rows={total} bbox={BBOX} days={DAY_RANGE}\n",
        encoding="utf-8",
    )
    print(f"DONE india_rows={total} dir={OUT_DIR}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
