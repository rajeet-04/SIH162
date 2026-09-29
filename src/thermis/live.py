"""Durable, single-worker FIRMS monitoring for the local advisory pilot."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np

from thermis.features import _distance_km
from thermis.weather import fetch_weather

PRODUCTS = ("VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "VIIRS_SNPP_NRT", "MODIS_NRT")


def utcnow():
    return datetime.now(timezone.utc)


def map_key():
    key = os.getenv("FIRMS_MAP_KEY", "").strip()
    path = Path(os.getenv("FIRMS_KEYS_FILE", "D:/data/firms_india/.map_keys"))
    if not key and path.is_file():
        key = next((line.strip() for line in path.read_text().splitlines() if line.strip()), "")
    if not key:
        raise ValueError("FIRMS key is not configured")
    return key


def fetch_product(product, start=None, days=2):
    # Never include a credential-bearing URL or response body in an exception/log.
    if product not in PRODUCTS or not 1 <= days <= 5:
        raise ValueError("Invalid FIRMS product or day range")
    suffix = f"/{start.isoformat()}" if start is not None else ""
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key()}/{product}/66,4,100,39/{days}{suffix}"
    try:
        with urlopen(url, timeout=30) as response:
            raw = response.read(20_000_001)
            if len(raw) > 20_000_000:
                raise ValueError("FIRMS response exceeds bounded download size")
            text = raw.decode("utf-8-sig")
    except HTTPError as error:
        raise RuntimeError(f"FIRMS HTTP {error.code}") from None
    except OSError:
        raise RuntimeError("FIRMS transport unavailable") from None
    reader = csv.DictReader(io.StringIO(text))
    required = {"latitude", "longitude", "acq_date", "acq_time", "frp"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("FIRMS response schema invalid or key rejected")
    return list(reader)


def normalize(row, product, now=None):
    now = now or utcnow()
    lat, lon, frp = (float(row[name]) for name in ("latitude", "longitude", "frp"))
    if not all(math.isfinite(x) for x in (lat, lon, frp)):
        raise ValueError("nonfinite measurement")
    if not (4 <= lat <= 39 and 66 <= lon <= 100 and frp >= 0):
        raise ValueError("measurement outside domain")
    stamp = datetime.strptime(
        f"{row['acq_date']} {str(row['acq_time']).zfill(4)}", "%Y-%m-%d %H%M"
    ).replace(tzinfo=timezone.utc)
    if stamp > now + timedelta(minutes=5):
        raise ValueError("future acquisition")
    brightness = row.get("bright_ti4", row.get("brightness"))
    brightness = float(brightness) if brightness not in (None, "") else None
    if brightness is not None and (not math.isfinite(brightness) or brightness <= 0):
        raise ValueError("invalid brightness")
    identity = f"{product}|{stamp.isoformat()}|{lat:.5f}|{lon:.5f}"
    return {
        "event_id": "firms-" + hashlib.sha256(identity.encode()).hexdigest()[:24],
        "latitude": lat,
        "longitude": lon,
        "timestamp_utc": stamp.isoformat(),
        "frp": frp,
        "brightness_temperature": brightness,
        "product": product,
        "satellite": row.get("satellite"),
        "source_confidence": row.get("confidence"),
        "daynight": row.get("daynight"),
    }


class EventStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY, timestamp TEXT NOT NULL,
                    latitude REAL, longitude REAL, observation TEXT NOT NULL,
                    payload TEXT, review TEXT);
                CREATE INDEX IF NOT EXISTS event_time ON events(timestamp);
                CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
                CREATE TABLE IF NOT EXISTS coverage (
                    product TEXT, day TEXT, status TEXT, rows INTEGER, checked_at TEXT,
                    PRIMARY KEY(product,day));
            """)
            columns = {r[1] for r in db.execute("PRAGMA table_info(events)")}
            if "history_only" not in columns:
                db.execute("ALTER TABLE events ADD COLUMN history_only INTEGER NOT NULL DEFAULT 0")
            db.execute(
                "CREATE INDEX IF NOT EXISTS event_location ON events(latitude,longitude,timestamp)"
            )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def observe(self, event):
        with self.connect() as db:
            return db.execute(
                "INSERT OR IGNORE INTO events(id,timestamp,latitude,longitude,observation) "
                "VALUES(?,?,?,?,?)",
                (
                    event["event_id"],
                    event["timestamp_utc"],
                    event["latitude"],
                    event["longitude"],
                    json.dumps(event, allow_nan=False),
                ),
            ).rowcount

    def pending(self):
        with self.connect() as db:
            return [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT observation FROM events WHERE payload IS NULL AND history_only=0 "
                    "ORDER BY timestamp DESC LIMIT 5000"
                )
            ]

    def save(self, event):
        with self.connect() as db:
            db.execute(
                "UPDATE events SET payload=? WHERE id=?",
                (json.dumps(event, allow_nan=False), event["event_id"]),
            )

    def events(self, limit=500):
        with self.connect() as db:
            rows = db.execute(
                "SELECT payload,review FROM events WHERE payload IS NOT NULL "
                "ORDER BY timestamp DESC,id LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(json.loads(p), review=json.loads(r) if r else None) for p, r in rows]

    def event(self, event_id):
        with self.connect() as db:
            row = db.execute("SELECT payload,review FROM events WHERE id=?", (event_id,)).fetchone()
        if not row or not row[0]:
            return None
        return dict(json.loads(row[0]), review=json.loads(row[1]) if row[1] else None)

    def review(self, event_id, value):
        with self.connect() as db:
            return db.execute(
                "UPDATE events SET review=? WHERE id=? AND payload IS NOT NULL",
                (json.dumps(value), event_id),
            ).rowcount

    def state(self, value=None):
        with self.connect() as db:
            if value is not None:
                db.execute(
                    "INSERT OR REPLACE INTO state VALUES('ingestion',?)", (json.dumps(value),)
                )
            row = db.execute("SELECT value FROM state WHERE key='ingestion'").fetchone()
        return json.loads(row[0]) if row else {}

    def history(self, event):
        stamp = datetime.fromisoformat(event["timestamp_utc"])
        with self.connect() as db:
            rows = db.execute(
                "SELECT timestamp,latitude,longitude FROM events WHERE timestamp>=? "
                "AND timestamp<? AND latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?",
                (
                    (stamp - timedelta(days=90)).isoformat(),
                    stamp.isoformat(),
                    event["latitude"] - 0.06,
                    event["latitude"] + 0.06,
                    event["longitude"] - 0.07,
                    event["longitude"] + 0.07,
                ),
            ).fetchall()
        nearby = []
        if rows:
            distances = _distance_km(
                event["latitude"],
                event["longitude"],
                np.array([r[1] for r in rows]),
                np.array([r[2] for r in rows]),
            )
            nearby = [datetime.fromisoformat(r[0]) for r, d in zip(rows, distances) if d <= 5]
        counts = {
            f"prior_detections_{d}d": sum(t >= stamp - timedelta(days=d) for t in nearby)
            for d in (7, 30, 90)
        }
        return dict(counts, stationary_count_90d=counts["prior_detections_90d"])

    def coverage(self, timestamp):
        """Coverage of the 90 completed UTC dates preceding an acquisition date."""
        end = datetime.fromisoformat(timestamp).astimezone(timezone.utc).date() - timedelta(days=1)
        start = end - timedelta(days=89)
        with self.connect() as db:
            complete = db.execute(
                "SELECT COUNT(*) FROM coverage WHERE day BETWEEN ? AND ? AND status='complete'",
                (str(start), str(end)),
            ).fetchone()[0]
        return dict(
            start=str(start),
            end=str(end),
            complete=complete == 90 * len(PRODUCTS),
            complete_product_days=complete,
            required_product_days=90 * len(PRODUCTS),
            scope="90 completed UTC dates; current acquisition day is provisional",
        )

    def timeline(self, event):
        """Observed daily counts within 5 km, anchored to the selected acquisition."""
        stamp = datetime.fromisoformat(event["timestamp_utc"])
        with self.connect() as db:
            rows = db.execute(
                "SELECT timestamp,latitude,longitude FROM events WHERE timestamp>=? "
                "AND timestamp<=? AND latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?",
                (
                    (stamp - timedelta(days=90)).isoformat(),
                    stamp.isoformat(),
                    event["latitude"] - 0.06,
                    event["latitude"] + 0.06,
                    event["longitude"] - 0.07,
                    event["longitude"] + 0.07,
                ),
            ).fetchall()
        if not rows:
            return []
        distances = _distance_km(
            event["latitude"],
            event["longitude"],
            np.array([r[1] for r in rows]),
            np.array([r[2] for r in rows]),
        )
        counts = {}
        for row, distance in zip(rows, distances, strict=True):
            if distance <= 5:
                days = (stamp.date() - datetime.fromisoformat(row[0]).date()).days
                counts[days] = counts.get(days, 0) + 1
        return [
            {"days_ago": days, "prior_detections": count}
            for days, count in sorted(counts.items(), reverse=True)
        ]


class LiveMonitor:
    def __init__(self, runtime, store, interval=300, fetcher=fetch_product, weather=fetch_weather):
        self.runtime, self.store = runtime, store
        self.interval = max(60, interval)
        self.fetcher, self.weather = fetcher, weather
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = None

    def status(self):
        value = self.store.state()
        with self.store.connect() as db:
            value["latest_acquisition_utc"] = db.execute(
                "SELECT MAX(timestamp) FROM events"
            ).fetchone()[0]
            value["pending_scores"] = db.execute(
                "SELECT COUNT(*) FROM events WHERE payload IS NULL AND history_only=0"
            ).fetchone()[0]
        last = value.get("last_success_utc")
        value["historical_coverage"] = self.store.coverage(utcnow().isoformat())
        stale = (
            not last
            or (utcnow() - datetime.fromisoformat(last)).total_seconds() > self.interval * 3
        )
        return dict(
            value, enabled=True, stale=stale, poll_seconds=self.interval, mode="live", advisory=True
        )

    def cycle(self):
        if not self.lock.acquire(blocking=False):
            return
        try:
            state = self.store.state()
            state.update(last_attempt_utc=utcnow().isoformat(), running=True, products={})
            self.store.state(state)
            added = rejected = duplicates = successful = 0
            for product in PRODUCTS:
                if self.stop_event.is_set():
                    break
                try:
                    fetch_day = utcnow().date()
                    rows = self.fetcher(product)
                    successful += 1
                    invalid = 0
                    for row in rows:
                        try:
                            new = self.store.observe(normalize(row, product))
                            added += new
                            duplicates += 1 - new
                        except (ValueError, TypeError, KeyError):
                            rejected += 1
                            invalid += 1
                    state["products"][product] = {
                        "status": "ok",
                        "rows": len(rows),
                        "rejected": invalid,
                    }
                    # Two-day overlap refreshes yesterday's ledger while the service runs.
                    # A request crossing UTC midnight cannot establish a fixed daily window.
                    if not invalid and utcnow().date() == fetch_day:
                        yesterday = str(fetch_day - timedelta(days=1))
                        count = sum(r.get("acq_date") == yesterday for r in rows)
                        with self.store.connect() as db:
                            db.execute(
                                "INSERT OR REPLACE INTO coverage VALUES(?,?,?,?,?)",
                                (product, yesterday, "complete", count, utcnow().isoformat()),
                            )
                except Exception as error:
                    state["products"][product] = {
                        "status": "unavailable",
                        "error": type(error).__name__,
                    }
            scored = failed = 0
            self.weather_budget = 12
            for observation in self.store.pending():
                if self.stop_event.is_set():
                    break
                try:
                    self.score(observation)
                    scored += 1
                except Exception:
                    failed += 1  # unscored rows stay durable and retry next cycle
            state.update(
                running=False,
                new_rows=added,
                rejected_rows=rejected,
                duplicate_rows=duplicates,
                scored_rows=scored,
                scoring_failures=failed,
                completed_at_utc=utcnow().isoformat(),
                next_poll_utc=(utcnow() + timedelta(seconds=self.interval)).isoformat(),
            )
            if successful:
                state["last_success_utc"] = utcnow().isoformat()
            state["status"] = "ok" if successful == len(PRODUCTS) and not failed else "degraded"
            self.store.state(state)
        finally:
            self.lock.release()

    def score(self, observation):
        from thermis.api import PredictionRequest

        history = self.store.history(observation)
        request = PredictionRequest(
            **{k: v for k, v in observation.items() if k in PredictionRequest.model_fields},
            **history,
        )
        prediction = self.runtime.predict(request)
        prediction.update(review_required=True, advisory=True)
        evidence = dict(
            observation,
            **history,
            history_scope="FIRMS observations with per-product daily backfill coverage",
            history_complete_90d=self.store.coverage(observation["timestamp_utc"])["complete"],
            historical_coverage=self.store.coverage(observation["timestamp_utc"]),
            nearest_flare_distance_m=None,
            nearest_industrial_distance_m=None,
            weather_status="unavailable",
            label_source="unlabeled live observation",
        )
        # Weather describes conditions now, not the historical overpass.
        try:
            if getattr(self, "weather_budget", 0) <= 0:
                raise RuntimeError("weather budget exhausted")
            self.weather_budget -= 1
            evidence["weather"] = self.weather(
                observation["latitude"], observation["longitude"]
            ).as_dict()
            evidence["weather_status"] = "current context; not acquisition-time weather"
        except Exception:
            pass
        event = {k: observation[k] for k in ("event_id", "latitude", "longitude", "timestamp_utc")}
        event.update(
            prediction=prediction,
            evidence=evidence,
            replay=False,
            ingested_at_utc=utcnow().isoformat(),
            timeline_90d=[],
        )
        self.store.save(event)

    def start(self):
        def run():
            while not self.stop_event.is_set():
                try:
                    self.cycle()
                except Exception:
                    self.store.state({"status": "error", "running": False})
                self.stop_event.wait(self.interval)

        self.thread = threading.Thread(target=run, name="firms-monitor", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=35)
