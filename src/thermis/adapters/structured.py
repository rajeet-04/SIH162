from pathlib import Path
from typing import Any

import pandas as pd

from thermis.contracts import ThermalEvent


def matlab_datenum_to_utc(value: int | float) -> pd.Timestamp:
    """Convert a MATLAB serial day number to a UTC timestamp."""
    return pd.Timestamp("1970-01-01", tz="UTC") + pd.to_timedelta(
        float(value) - 719529, unit="D"
    )


def _timestamp_to_utc(value: Any) -> pd.Timestamp:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if 60_000 <= number <= 1_000_000:
            return matlab_datenum_to_utc(number)
        unit = "ns" if abs(number) >= 1e14 else "ms"
        return pd.to_datetime(number, unit=unit, utc=True)
    return pd.to_datetime(value, utc=True)


def make_event_id(source_name: str, source_file: str, row_number: int) -> str:
    return f"{source_name}:{source_file}:{row_number}"


def normalize_frp_frame(raw: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Normalize an FRP-like frame and retain invalid rows in a reject column."""
    records: list[dict[str, Any]] = []
    for row_number, row in raw.reset_index(drop=True).iterrows():
        source_file = str(row.get("source_file", "unknown"))
        record: dict[str, Any] = {
            "event_id": make_event_id(source_name, source_file, row_number),
            "source_dataset": source_name,
            "source_record_id": str(row_number),
            "latitude": row.get("latitude"),
            "longitude": row.get("longitude"),
            "timestamp_utc": None,
            "brightness_temperature": row.get("BT_MIR"),
            "frp": row.get("FRP_MWIR"),
            "frp_uncertainty": row.get("FRP_uncertainty_MWIR"),
            "zone": row.get("zone"),
            "source_file": source_file,
            "reject_reason": None,
        }
        try:
            record["timestamp_utc"] = _timestamp_to_utc(row["time"])
            event = ThermalEvent.model_validate(record)
            record.update(event.model_dump())
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            record["reject_reason"] = str(error)
        records.append(record)
    return pd.DataFrame(records)


def read_frp_events(path: Path) -> pd.DataFrame:
    return normalize_frp_frame(pd.read_csv(path), source_name="frp")


def read_flare_points(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    return normalize_frp_frame(raw, source_name="flaresat")
