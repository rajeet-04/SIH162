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
    "nearest_flare_distance_m",
    # nearest_industrial_distance_m deliberately excluded: build_event_features
    # populates it from the same gas-flaring points, so it duplicates
    # nearest_flare_distance_m exactly (ponytail: restore when a real OSM
    # industrial-facility source lands).
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
                np.count_nonzero((timestamps[:position] >= cutoff)
                                 & (timestamps[:position] < current_time)
                                 & (distance_km <= radius_km))
            )
        stationary[position] = int(
            np.count_nonzero((timestamps[:position] >= current_time - np.timedelta64(90, "D"))
                             & (timestamps[:position] < current_time)
                             & (distance_km <= radius_km))
        )
    for name, values in outputs.items():
        ordered[name] = values
    ordered["stationary_count_90d"] = stationary
    return ordered.sort_index()


def build_event_features(events: pd.DataFrame, context: object | None = None) -> pd.DataFrame:
    """Build the fixed MVP feature set and preserve feature-time provenance."""
    result = add_persistence_features(events)
    for column in ("frp", "brightness_temperature", "frp_uncertainty"):
        if column not in result:
            result[column] = np.nan
    result["feature_as_of_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True)
    result["nearest_flare_distance_m"] = np.nan
    result["nearest_industrial_distance_m"] = np.nan
    for kind in ("flare", "industrial"):
        if not isinstance(context, dict) or not isinstance(context.get(f"{kind}_points"),
                                                          pd.DataFrame):
            continue
        points = context[f"{kind}_points"]
        valid = points[["latitude", "longitude"]].dropna()
        point_lat = valid["latitude"].to_numpy(dtype=float)
        point_lon = valid["longitude"].to_numpy(dtype=float)
        for index, row in result.iterrows():
            distances = _distance_km(
                float(row["latitude"]), float(row["longitude"]), point_lat, point_lon
            )
            nearest_m = float(np.min(distances) * 1000) if len(distances) else np.nan
            result.at[index, f"nearest_{kind}_distance_m"] = nearest_m
    return result
