import pandas as pd

from thermis.features import add_persistence_features


def test_future_events_do_not_change_past_persistence() -> None:
    base = pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "timestamp_utc": pd.to_datetime(["2024-01-01", "2024-01-10"], utc=True),
            "latitude": [20.0, 20.0],
            "longitude": [70.0, 70.0],
        }
    )
    first = add_persistence_features(base).set_index("event_id")
    extended = pd.concat(
        [
            base,
            pd.DataFrame(
                {
                    "event_id": ["future"],
                    "timestamp_utc": pd.to_datetime(["2024-04-01"], utc=True),
                    "latitude": [20.0],
                    "longitude": [70.0],
                }
            ),
        ],
        ignore_index=True,
    )
    second = add_persistence_features(extended).set_index("event_id")
    assert first.loc["a", "prior_detections_90d"] == second.loc["a", "prior_detections_90d"] == 0
