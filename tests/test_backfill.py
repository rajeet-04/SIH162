from datetime import date, datetime, timezone

from thermis.live import PRODUCTS, EventStore, normalize


def test_backfill_resumes_and_keeps_history_out_of_live_scoring(tmp_path):
    from thermis.backfill import backfill

    store = EventStore(tmp_path / "db")

    def fetch(product, start, days):
        return [
            dict(
                latitude="20",
                longitude="75",
                frp="10",
                bright_ti4="320",
                acq_date=str(start),
                acq_time="1200",
            )
        ]

    availability = {p: ("2026-01-01", "2026-09-06") for p in PRODUCTS}
    result = backfill(store, date(2026, 9, 4), date(2026, 9, 5), fetch, availability)
    assert result["complete_product_days"] == 8
    assert store.pending() == []
    target = normalize(
        dict(latitude="20", longitude="75", frp="10", acq_date="2026-09-06", acq_time="1200"),
        PRODUCTS[0],
        datetime(2026, 9, 7, tzinfo=timezone.utc),
    )
    assert store.history(target)["prior_detections_7d"] == 4

    def unavailable(*args):
        raise AssertionError("completed chunks must not be fetched again")

    again = backfill(store, date(2026, 9, 4), date(2026, 9, 5), unavailable, availability)
    assert again["complete_product_days"] == 8
    assert store.history(target)["prior_detections_7d"] == 4


def test_missing_or_invalid_days_never_count_as_complete(tmp_path):
    from thermis.backfill import backfill

    store = EventStore(tmp_path / "db")
    availability = {p: ("2026-09-05", "2026-09-06") for p in PRODUCTS}

    def fetch(*args):
        return [
            dict(latitude="NaN", longitude="75", frp="1", acq_date="2026-09-05", acq_time="1200")
        ]

    result = backfill(store, date(2026, 9, 4), date(2026, 9, 5), fetch, availability)
    assert result["complete_product_days"] == 0
    assert result["required_product_days"] == 8
    assert not result["complete"]


def test_coverage_requires_every_product_and_day(tmp_path):
    from thermis.backfill import backfill

    store = EventStore(tmp_path / "db")
    availability = {p: ("2026-01-01", "2026-09-06") for p in PRODUCTS}
    backfill(store, date(2026, 6, 8), date(2026, 9, 5), lambda *a: [], availability)
    assert store.coverage("2026-09-06T12:00:00+00:00")["complete"]
    assert not store.coverage("2026-09-05T12:00:00+00:00")["complete"]


def test_refresh_preserves_review_ingestion_and_weather(tmp_path):
    from thermis.backfill import refresh_live
    from thermis.live import LiveMonitor

    class Runtime:
        def predict(self, request):
            return {"final_class": "test", "confidence": 0.5}

    store = EventStore(tmp_path / "db")
    observation = normalize(
        dict(latitude="20", longitude="75", frp="10", acq_date="2026-09-05", acq_time="1200"),
        PRODUCTS[0],
    )
    store.observe(observation)
    monitor = LiveMonitor(Runtime(), store)
    monitor.score(observation)
    previous = store.event(observation["event_id"])
    previous["evidence"]["weather"] = {"temperature_c": 30}
    previous["evidence"]["weather_status"] = "current context"
    store.save(previous)
    store.review(observation["event_id"], {"decision": "dismissed"})
    assert refresh_live(monitor) == {"refreshed": 1, "failed": 0}
    updated = store.event(observation["event_id"])
    assert updated["review"]["decision"] == "dismissed"
    assert updated["ingested_at_utc"] == previous["ingested_at_utc"]
    assert updated["evidence"]["weather"] == {"temperature_c": 30}
