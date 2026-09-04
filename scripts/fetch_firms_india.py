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
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MAP_KEY = os.environ.get("FIRMS_MAP_KEY", "").strip()
VERSION = os.environ.get("FIRMS_VERSION", "4.0")
DAY_RANGE = os.environ.get("FIRMS_DAY_RANGE", "10")
OUT_DIR = Path(os.environ.get("FIRMS_OUT_DIR", r"D:/data/firms_india"))
BBOX = "66,4,100,39"  # west,south,east,north
PRODUCTS = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "MODIS_NRT"]


def fetch_product(product: str) -> pd.DataFrame:
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{VERSION}/{MAP_KEY}/{product}/{BBOX}/{DAY_RANGE}"
    )
    print(f"GET {product}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            text = response.read().decode("utf-8", errors="replace")
    except Exception as error:  # network or HTTP failure
        raise SystemExit(f"FIRMS request failed for {product}: {error}")
    if text.startswith(("Invalid", "Error", "<")):
        raise SystemExit(f"FIRMS rejected {product}: {text.strip()[:200]}")
    path = OUT_DIR / f"firms_{product.lower()}_india_raw.csv"
    path.write_text(text, encoding="utf-8")
    frame = pd.read_csv(path)
    print(f"  {len(frame)} rows -> {path.name}", flush=True)
    return frame


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
    out = pd.DataFrame(
        {
            "latitude": frame["latitude"],
            "longitude": frame["longitude"],
            "time": frame["time"],
            "BT_MIR": brightness,
            "FRP_MWIR": frame.get("frp"),
            "FRP_uncertainty_MWIR": float("nan"),
            "source_file": f"firms_{product.lower()}_india.csv",
            "firms_confidence": frame.get("confidence"),
            "firms_satellite": frame.get("satellite"),
            "firms_daynight": frame.get("daynight"),
        }
    )
    return out.dropna(subset=["latitude", "longitude", "time"])


def main() -> None:
    if not MAP_KEY:
        raise SystemExit("Set FIRMS_MAP_KEY first (see module docstring).")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for product in PRODUCTS:
        raw = fetch_product(product)
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
