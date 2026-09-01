import pandas as pd

from thermis.labels import assign_event_label


def test_repeated_stationary_known_flare_is_persistent_heat() -> None:
    row = pd.Series(
        {
            "prior_detections_90d": 42,
            "nearest_flare_distance_m": 180.0,
            "modis_burned_overlap": False,
            "spread_rate_km_h": 0.0,
            "nearest_industrial_distance_m": 150.0,
        }
    )
    label = assign_event_label(row)
    assert label.stage_1 == "persistent_or_non_emergency_heat"
    assert label.stage_2 == "persistent_industrial_heat_or_flare"
    assert label.confidence >= 0.8


def test_generic_fire_source_is_not_automatically_industrial() -> None:
    row = pd.Series(
        {
            "source_label": "fire",
            "nearest_industrial_distance_m": 8000.0,
            "modis_burned_overlap": True,
            "spread_rate_km_h": 0.7,
        }
    )
    assert assign_event_label(row).stage_2 == "wildfire_or_agricultural_burn"


def test_label_provenance_is_parquet_serializable(tmp_path) -> None:
    from thermis.labels import assign_labels

    labeled = assign_labels(pd.DataFrame([{"event_id": "e1"}]))
    labeled.to_parquet(tmp_path / "labels.parquet", index=False)
