"""Small, dependency-free Open-Meteo enrichment client.

Weather is advisory context in the current model release. It is deliberately
not silently injected into the trained feature vector until a retrained model
has been evaluated with the same fields.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from urllib.parse import urlencode
from urllib.request import Request, urlopen

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


@dataclass(frozen=True)
class WeatherContext:
    latitude: float
    longitude: float
    temperature_c: float | None
    relative_humidity_pct: float | None
    wind_speed_kmh: float | None
    wind_direction_deg: float | None
    precipitation_mm: float | None
    cloud_cover_pct: float | None
    observed_at_utc: str | None
    source: str = "open-meteo"

    def as_dict(self) -> dict[str, object]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "temperature_c": self.temperature_c,
            "relative_humidity_pct": self.relative_humidity_pct,
            "wind_speed_kmh": self.wind_speed_kmh,
            "wind_direction_deg": self.wind_direction_deg,
            "precipitation_mm": self.precipitation_mm,
            "cloud_cover_pct": self.cloud_cover_pct,
            "observed_at_utc": self.observed_at_utc,
            "source": self.source,
        }


_cache: dict[tuple[float, float], tuple[float, WeatherContext]] = {}


def fetch_weather(latitude: float, longitude: float, *, timeout: float = 8.0) -> WeatherContext:
    """Fetch current weather for a point, with a short in-process cache."""
    key = (round(latitude, 2), round(longitude, 2))
    cached = _cache.get(key)
    if cached and time.monotonic() - cached[0] < 600:
        return cached[1]

    query = urlencode(
        {
            "latitude": key[0],
            "longitude": key[1],
            "current": (
                "temperature_2m,relative_humidity_2m,wind_speed_10m,"
                "wind_direction_10m,precipitation,cloud_cover"
            ),
            "timezone": "UTC",
        }
    )
    request = Request(
        f"{OPEN_METEO_URL}?{query}",
        headers={"User-Agent": "THERMIS-SIH26162/0.1"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    current = payload.get("current", {})
    context = WeatherContext(
        latitude=key[0],
        longitude=key[1],
        temperature_c=_number(current.get("temperature_2m")),
        relative_humidity_pct=_number(current.get("relative_humidity_2m")),
        wind_speed_kmh=_number(current.get("wind_speed_10m")),
        wind_direction_deg=_number(current.get("wind_direction_10m")),
        precipitation_mm=_number(current.get("precipitation")),
        cloud_cover_pct=_number(current.get("cloud_cover")),
        observed_at_utc=current.get("time"),
    )
    _cache[key] = (time.monotonic(), context)
    return context


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None
