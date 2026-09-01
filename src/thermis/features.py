from typing import Final

import numpy as np
import pandas as pd

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "latitude",
    "longitude",
    "frp",
    "brightness_temperature",
    "frp_uncertainty",
    "prior_detections_7d",
    "prior_detections_30d",
    "prior_detections_90d",
    "stationary_count_90d",
)


def _distance_km(
    latitude: float, longitude: float, prior_lat: np.ndarray, prior_lon: np.ndarray
) -> np.ndarray:
    earth_radius_km = 6371.0088
    lat1 = np.radians(latitude)
    lat2 = np.radians(prior_lat)
    dlat = lat2 - lat1
    dlon = np.radians(prior_lon - longitude)
    haversine = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * earth_radius_km * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))


def add_persistence_features(
    frame: pd.DataFrame, radius_km: float = 5.0, windows_days: tuple[int, ...] = (7, 30, 90)
) -> pd.DataFrame:
    """Add strictly historical nearby-event counts for each event."""
    result = frame.copy()
    result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True)
    ordered = result.sort_values("timestamp_utc")
    timestamps = ordered["timestamp_utc"].to_numpy()
    latitudes = ordered["latitude"].to_numpy(dtype=float)
    longitudes = ordered["longitude"].to_numpy(dtype=float)
    outputs = {
        f"prior_detections_{days}d": np.zeros(len(ordered), dtype=int) for days in windows_days
    }
    stationary = np.zeros(len(ordered), dtype=int)
    for position in range(len(ordered)):
        current_time = timestamps[position]
        distance_km = _distance_km(
            latitudes[position], longitudes[position], latitudes[:position], longitudes[:position]
        )
        for days in windows_days:
            cutoff = current_time - np.timedelta64(days, "D")
            outputs[f"prior_detections_{days}d"][position] = int(
                np.count_nonzero((timestamps[:position] >= cutoff) & (distance_km <= radius_km))
            )
        stationary[position] = int(
            np.count_nonzero((timestamps[:position] >= current_time - np.timedelta64(90, "D"))
                             & (distance_km <= radius_km))
        )
    for name, values in outputs.items():
        ordered[name] = values
    ordered["stationary_count_90d"] = stationary
    return ordered.sort_index()


def build_event_features(events: pd.DataFrame, context: object | None = None) -> pd.DataFrame:
    """Build the fixed MVP feature set and preserve feature-time provenance."""
    del context
    result = add_persistence_features(events)
    for column in ("frp", "brightness_temperature", "frp_uncertainty"):
        if column not in result:
            result[column] = np.nan
    result["feature_as_of_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True)
    return result
