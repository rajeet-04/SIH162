from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from thermis.api import create_app
from thermis.live import PRODUCTS, EventStore, LiveMonitor, normalize


def row(stamp=None, **changes):
    stamp = stamp or datetime.now(timezone.utc) - timedelta(hours=2)
    return dict(
        latitude="20.0",
        longitude="75.0",
        frp="15.0",
        bright_ti4="320",
        acq_date=str(stamp.date()),
        acq_time=stamp.strftime("%H%M"),
        **changes,
    )


class Runtime:
    def predict(self, request):
        return {
            "final_class": "industrial_fire_candidate",
            "confidence": 0.6,
            "risk_score": 40,
            "risk_level": "moderate",
            "probabilities": {},
            "review_required": False,
            "model_version": "test",
        }


def no_weather(*args):
    raise OSError("offline")


def test_restart_deduplicates_and_preserves_review(tmp_path):
    path = tmp_path / "events.sqlite3"
    store = EventStore(path)
    monitor = LiveMonitor(Runtime(), store, fetcher=lambda _: [row()], weather=no_weather)
    monitor.cycle()
    events = store.events()
    assert len(events) == len(PRODUCTS)
    assert all(e["prediction"]["review_required"] and not e["replay"] for e in events)
    store.review(events[0]["event_id"], {"decision": "dismissed"})
    restarted = LiveMonitor(
        Runtime(), EventStore(path), fetcher=lambda _: [row()], weather=no_weather
    )
    restarted.cycle()
    assert len(restarted.store.events()) == len(PRODUCTS)
    assert restarted.store.state()["duplicate_rows"] == len(PRODUCTS)
    assert restarted.store.event(events[0]["event_id"])["review"]["decision"] == "dismissed"


@pytest.mark.parametrize(
    "field,value", [("latitude", "NaN"), ("longitude", "200"), ("frp", "-1"), ("bright_ti4", "inf")]
)
def test_reject_invalid_observations(field, value):
    data = row()
    data[field] = value
    with pytest.raises(ValueError):
        normalize(data, PRODUCTS[0])


def test_reject_future_observation():
    with pytest.raises(ValueError):
        normalize(row(datetime.now(timezone.utc) + timedelta(days=1)), PRODUCTS[0])


def test_history_excludes_current_same_time_future_and_distant(tmp_path):
    store = EventStore(tmp_path / "db")
    now = datetime.now(timezone.utc) - timedelta(days=1)
    target = normalize(row(now), PRODUCTS[0])
    for stamp, lat in [
        (now - timedelta(days=1), "20"),
        (now, "20"),
        (now + timedelta(hours=1), "20"),
        (now - timedelta(days=1), "25"),
    ]:
        data = row(stamp)
        data["latitude"] = lat
        store.observe(normalize(data, PRODUCTS[0]))
    assert store.history(target)["prior_detections_7d"] == 1


def test_outage_preserves_events_and_reports_degraded(tmp_path):
    store = EventStore(tmp_path / "db")
    monitor = LiveMonitor(Runtime(), store, fetcher=lambda _: [row()], weather=no_weather)
    monitor.cycle()

    def fail(_):
        raise OSError("secret must not appear")

    monitor.fetcher = fail
    monitor.cycle()
    assert len(store.events()) == len(PRODUCTS)
    assert monitor.status()["status"] == "degraded"
    assert "secret" not in str(monitor.status())


def test_live_api_and_review_contract(tmp_path):
    store = EventStore(tmp_path / "db")
    monitor = LiveMonitor(Runtime(), store, fetcher=lambda _: [row()], weather=no_weather)
    monitor.cycle()
    client = TestClient(create_app(Runtime(), monitor))
    data = client.get("/events").json()
    assert data["mode"] == "live"
    event_id = data["events"][0]["event_id"]
    assert client.get(f"/events/{event_id}/evidence").status_code == 200
    assert (
        client.post(f"/events/{event_id}/review", json={"decision": "dismissed"}).status_code == 200
    )
    assert client.get(f"/events/{event_id}").json()["review"]["decision"] == "dismissed"
    assert (
        client.post(f"/events/{event_id}/review", json={"decision": "invented"}).status_code == 422
    )


def test_failed_scoring_retries_without_refetch_loss(tmp_path):
    store = EventStore(tmp_path / "db")

    class Broken:
        def predict(self, request):
            raise ValueError("model failure")

    monitor = LiveMonitor(Broken(), store, fetcher=lambda _: [row()], weather=no_weather)
    monitor.cycle()
    assert not store.events()
    assert len(store.pending()) == len(PRODUCTS)
    monitor.runtime = Runtime()
    monitor.cycle()
    assert len(store.events()) == len(PRODUCTS)


def test_timeline_uses_only_observed_nearby_days_before_target(tmp_path):
    store = EventStore(tmp_path / "db")
    stamp = datetime.now(timezone.utc) - timedelta(days=2)
    target = normalize(row(stamp), PRODUCTS[0])
    store.observe(target)
    store.observe(normalize(row(stamp - timedelta(days=1)), PRODUCTS[0]))
    store.observe(normalize(row(stamp + timedelta(days=1)), PRODUCTS[0]))
    far = row(stamp - timedelta(days=1))
    far["latitude"] = "25"
    store.observe(normalize(far, PRODUCTS[0]))
    assert store.timeline(target) == [
        {"days_ago": 1, "prior_detections": 1},
        {"days_ago": 0, "prior_detections": 1},
    ]


def test_live_overlap_records_yesterday_coverage(tmp_path):
    store = EventStore(tmp_path / "db")
    monitor = LiveMonitor(Runtime(), store, fetcher=lambda _: [], weather=no_weather)
    monitor.cycle()
    with store.connect() as db:
        rows = db.execute("SELECT product,day FROM coverage WHERE status='complete'").fetchall()
    yesterday = str((datetime.now(timezone.utc) - timedelta(days=1)).date())
    assert set(rows) == {(p, yesterday) for p in PRODUCTS}
