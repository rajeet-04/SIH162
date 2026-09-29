"""Interactive Earthdata token setup and bounded protected-TIFF access verification."""

import getpass
import json
from pathlib import Path
from urllib.parse import urlparse

import httpx
from dotenv import set_key


def trusted_download(url):
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.hostname == "data.lpdaac.earthdatacloud.nasa.gov"
        and parsed.path.startswith("/lp-prod-protected/")
        and parsed.path.endswith(".tif")
    )


def verify_bytes(data, content_type):
    return "html" not in content_type.lower() and data[:4] in (
        b"II*\x00",
        b"MM\x00*",
        b"II+\x00",
        b"MM\x00+",
    )


def persist_token(value, path):
    token = value.get("access_token")
    if not isinstance(token, str) or not token.strip() or any(c.isspace() for c in token):
        raise ValueError("Earthdata did not return a valid token")
    set_key(str(path), "NASA_EARTHDATA_TOKEN", token, quote_mode="always")
    return token


def verify_download(client, url, token):
    if not trusted_download(url):
        raise ValueError("Untrusted protected download origin")
    source = url
    headers = {"Authorization": f"Bearer {token}", "Range": "bytes=0-4095"}
    for _ in range(4):
        with client.stream("GET", url, headers=headers, follow_redirects=False) as download:
            if download.status_code in (301, 302, 303, 307, 308):
                target = download.headers.get("location", "")
                parsed = urlparse(target)
                host = parsed.hostname or ""
                if parsed.scheme != "https" or not (
                    host.endswith(".cloudfront.net") or host.endswith(".amazonaws.com")
                ):
                    raise ValueError(
                        "Protected download redirected outside approved delivery hosts"
                    )
                url = target
                headers = {"Range": "bytes=0-4095"}
                continue
            data = b""
            if download.status_code in (200, 206):
                for chunk in download.iter_bytes(chunk_size=4096):
                    data += chunk[: 4096 - len(data)]
                    if len(data) >= 4096:
                        break
            return dict(
                protected_download_verified=verify_bytes(
                    data, download.headers.get("content-type", "")
                ),
                http_status=download.status_code,
                bytes_verified=len(data),
                source=source,
                note="Bounded TIFF header access test; not a complete image download",
            )
    raise ValueError("Protected download exceeded redirect limit")


def main():
    username = input("Earthdata username: ").strip()
    password = getpass.getpass("Earthdata password (not saved): ")
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        response = client.post(
            "https://urs.earthdata.nasa.gov/api/users/find_or_create_token",
            auth=(username, password),
        )
        del password
        if response.status_code != 200:
            print(json.dumps(dict(authenticated=False, http_status=response.status_code)))
            return
        value = response.json()
        token = persist_token(value, Path(".env"))
        print(
            json.dumps(
                dict(
                    authenticated=True,
                    token_saved=True,
                    expiration_date=value.get("expiration_date"),
                )
            ),
            flush=True,
        )
        response = client.get(
            "https://cmr.earthdata.nasa.gov/search/granules.json",
            params={"short_name": "HLSL30", "page_size": 5, "sort_key": "-start_date"},
        )
        response.raise_for_status()
        links = [
            link["href"]
            for entry in response.json()["feed"]["entry"]
            for link in entry.get("links", [])
            if trusted_download(link.get("href", ""))
        ]
        if not links:
            print(
                json.dumps(
                    dict(
                        protected_download_verified=False,
                        reason="No protected TIFF link discovered",
                    )
                )
            )
            return
        report = verify_download(client, links[0], token)
        Path("data/live").mkdir(parents=True, exist_ok=True)
        Path("data/live/nasa-access.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Neither credential-bearing request objects nor response bodies are logged.
        print(json.dumps(dict(status="failed", error=type(error).__name__)))
