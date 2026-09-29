"""On-demand evidence; never silently changes the frozen model feature contract."""

import json
import os
from datetime import datetime, timedelta
from functools import lru_cache
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@lru_cache(maxsize=256)
def satellite_evidence(latitude, longitude, date):
    stamp = datetime.fromisoformat(date)
    query = urlencode(
        {
            "short_name": "HLSL30",
            "page_size": 5,
            "point": f"{longitude},{latitude}",
            "temporal": f"{(stamp - timedelta(days=3)).date()}T00:00:00Z,"
            f"{(stamp + timedelta(days=1)).date()}T23:59:59Z",
            "sort_key": "-start_date",
        }
    )
    headers = {"User-Agent": "THERMIS/0.2"}
    token = os.getenv("NASA_EARTHDATA_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        f"https://cmr.earthdata.nasa.gov/search/granules.json?{query}", headers=headers
    )
    with urlopen(request, timeout=15) as response:
        entries = json.load(response).get("feed", {}).get("entry", [])
    return {
        "source": "NASA CMR / HLSL30",
        "token_configured": bool(token),
        "note": "Scene discovery only; imagery has not been verified by the image model.",
        "scenes": [
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "time_start": item.get("time_start"),
                "url": "https://search.earthdata.nasa.gov/search?"
                + urlencode({"q": item.get("title", "")}),
            }
            for item in entries
        ],
    }


@lru_cache(maxsize=256)
def osm_evidence(latitude, longitude, day):
    # Daily cache; OSM is incomplete and proximity alone does not establish a fire's cause.
    around = f"(around:5000,{float(latitude)},{float(longitude)})"
    query = (
        f'[out:json][timeout:10];(nwr["landuse"="industrial"]{around};'
        f'nwr["industrial"]{around};nwr["man_made"="flare"]{around};);out center 30;'
    )
    request = Request(
        "https://overpass-api.de/api/interpreter",
        data=urlencode({"data": query}).encode(),
        headers={"User-Agent": "THERMIS/0.2"},
    )
    with urlopen(request, timeout=15) as response:
        entries = json.load(response).get("elements", [])
    return {
        "source": "OpenStreetMap contributors",
        "as_of": day,
        "note": "Within 5 km; incomplete OSM coverage; contextual evidence only.",
        "facilities": [
            {
                "name": e.get("tags", {}).get("name", "Unnamed facility"),
                "tags": e.get("tags", {}),
                "url": f"https://www.openstreetmap.org/{e['type']}/{e['id']}",
            }
            for e in entries
        ],
    }
