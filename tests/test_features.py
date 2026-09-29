import pandas as pd

from thermis.features import add_persistence_features


def test_prior_nearby_count_excludes_current_event() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "timestamp_utc": pd.to_datetime(["2024-01-01", "2024-01-10"], utc=True),
            "latitude": [20.0, 20.0],
            "longitude": [70.0, 70.0],
        }
    )
    result = add_persistence_features(frame).set_index("event_id")
    assert result.loc["a", "prior_detections_90d"] == 0
    assert result.loc["b", "prior_detections_90d"] == 1
