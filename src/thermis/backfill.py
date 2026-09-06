"""Resumable FIRMS history import; historical rows do not flood live scoring."""

import csv
import io
import json
from datetime import date, timedelta
from urllib.request import urlopen

from thermis.live import PRODUCTS, EventStore, fetch_product, map_key, normalize, utcnow


def availability():
    try:
        with urlopen(
            "https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/" + map_key() + "/ALL",
            timeout=30,
        ) as response:
            rows = csv.DictReader(io.StringIO(response.read(100_000).decode("utf-8-sig")))
            return {r["data_id"]: (r["min_date"], r["max_date"]) for r in rows}
    except Exception:
        raise RuntimeError("FIRMS availability lookup failed") from None


def backfill(store, start, end, fetcher=fetch_product, available=None, progress=None):
    if start > end or end >= utcnow().date():
        raise ValueError("Backfill requires completed UTC dates in ascending order")
    available = availability() if available is None else available
    for product in PRODUCTS:
        day = start
        while day <= end:
            with store.connect() as db:
                done = db.execute(
                    "SELECT status FROM coverage WHERE product=? AND day=?", (product, str(day))
                ).fetchone()
            if done and done[0] == "complete":
                day += timedelta(days=1)
                continue
            low, high = available.get(product, ("9999-12-31", "0001-01-01"))
            if not low <= str(day) <= high:
                with store.connect() as db:
                    db.execute(
                        "INSERT OR REPLACE INTO coverage VALUES(?,?,?,?,?)",
                        (product, str(day), "unavailable", 0, utcnow().isoformat()),
                    )
                day += timedelta(days=1)
                continue
            last = min(day + timedelta(days=4), end, date.fromisoformat(high))
            counts = {str(day + timedelta(days=i)): 0 for i in range((last - day).days + 1)}
            added = 0
            status = "complete"
            try:
                rows = fetcher(product, day, (last - day).days + 1)
                events = []
                for row in rows:
                    event = normalize(row, product)
                    event_day = event["timestamp_utc"][:10]
                    if event_day not in counts:
                        raise ValueError("Observation outside requested dates")
                    counts[event_day] += 1
                    events.append(event)
                with store.connect() as db:
                    before = db.total_changes
                    db.executemany(
                        "INSERT OR IGNORE INTO events"
                        "(id,timestamp,latitude,longitude,observation,history_only) "
                        "VALUES(?,?,?,?,?,1)",
                        [
                            (
                                e["event_id"],
                                e["timestamp_utc"],
                                e["latitude"],
                                e["longitude"],
                                json.dumps(e, allow_nan=False),
                            )
                            for e in events
                        ],
                    )
                    added = db.total_changes - before
                    db.executemany(
                        "INSERT OR REPLACE INTO coverage VALUES(?,?,?,?,?)",
                        [(product, d, status, n, utcnow().isoformat()) for d, n in counts.items()],
                    )
            except Exception as error:
                status = "failed:" + type(error).__name__
                with store.connect() as db:
                    for d in counts:
                        # A failed retry cannot downgrade previously verified days.
                        db.execute(
                            "INSERT INTO coverage VALUES(?,?,?,?,?) ON CONFLICT(product,day) "
                            "DO UPDATE SET status=excluded.status "
                            "WHERE coverage.status!='complete'",
                            (product, d, status, 0, utcnow().isoformat()),
                        )
            if progress:
                progress(
                    dict(
                        product=product,
                        start=str(day),
                        end=str(last),
                        status=status,
                        new_rows=added,
                    )
                )
            day = last + timedelta(days=1)
    with store.connect() as db:
        completed = db.execute(
            "SELECT COUNT(*) FROM coverage WHERE day BETWEEN ? AND ? AND status='complete'",
            (str(start), str(end)),
        ).fetchone()[0]
    required = ((end - start).days + 1) * len(PRODUCTS)
    return dict(
        start=str(start),
        end=str(end),
        complete_product_days=completed,
        required_product_days=required,
        complete=completed == required,
    )


def refresh_live(monitor):
    """Refresh existing live scores in bounded batches, retaining review and context."""
    refreshed = failed = 0
    cursor = ""
    while True:
        with monitor.store.connect() as db:
            rows = db.execute(
                "SELECT id,observation FROM events WHERE history_only=0 AND id>? "
                "ORDER BY id LIMIT 200",
                (cursor,),
            ).fetchall()
        if not rows:
            break
        for event_id, observation in rows:
            cursor = event_id
            previous = monitor.store.event(event_id)
            try:
                monitor.score(json.loads(observation))
                if previous:
                    updated = monitor.store.event(event_id)
                    updated["ingested_at_utc"] = previous["ingested_at_utc"]
                    updated["rescored_at_utc"] = utcnow().isoformat()
                    for key in ("weather", "weather_status"):
                        if key in previous["evidence"]:
                            updated["evidence"][key] = previous["evidence"][key]
                    monitor.store.save(updated)
                refreshed += 1
            except Exception:
                failed += 1
    return dict(refreshed=refreshed, failed=failed)


def main():
    import argparse
    from pathlib import Path

    from dotenv import load_dotenv

    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    store = EventStore("data/live/events.sqlite3")
    end = utcnow().date() - timedelta(days=1)
    result = backfill(
        store,
        end - timedelta(days=89),
        end,
        progress=lambda value: print(json.dumps(value), flush=True),
    )
    print(json.dumps(result), flush=True)
    if args.refresh and result["complete"]:
        from thermis.api import ModelRuntime
        from thermis.live import LiveMonitor

        runtime = ModelRuntime.from_paths(Path("models/tabular"))
        print(json.dumps(refresh_live(LiveMonitor(runtime, store))), flush=True)


if __name__ == "__main__":
    main()
