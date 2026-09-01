from pathlib import Path

import pandas as pd

from thermis.adapters.structured import matlab_datenum_to_utc, normalize_frp_frame


def test_normalize_frp_frame_uses_utc_and_valid_coordinates() -> None:
    raw = pd.DataFrame(
        {
            "latitude": [22.5],
            "longitude": [41.7],
            "time": [747342048925155],
            "FRP_MWIR": [2.85],
            "FRP_uncertainty_MWIR": [0.30],
            "BT_MIR": [307.25],
            "zone": ["gulf_flaring"],
            "source_file": ["scene.zip"],
        }
    )
    result = normalize_frp_frame(raw, source_name="sentinel_frp")
    assert result.loc[0, "timestamp_utc"].tzinfo is not None
    assert result.loc[0, "event_id"].startswith("sentinel_frp:")
    assert result.loc[0, "frp"] == 2.85
    assert result.loc[0, "reject_reason"] is None


def test_matlab_datenum_epoch_is_utc() -> None:
    assert matlab_datenum_to_utc(719529).isoformat() == "1970-01-01T00:00:00+00:00"


def test_invalid_coordinates_are_rejected_without_silent_drop() -> None:
    raw = pd.DataFrame(
        {
            "latitude": [95.0],
            "longitude": [41.7],
            "time": [747342048925155],
            "FRP_MWIR": [2.85],
            "FRP_uncertainty_MWIR": [0.30],
            "BT_MIR": [307.25],
            "source_file": ["bad.csv"],
        }
    )
    result = normalize_frp_frame(raw, source_name="sentinel_frp")
    assert len(result) == 1
    assert result.loc[0, "reject_reason"]


def test_fixture_path_is_a_path_object() -> None:
    assert Path("tests/fixtures/frp.csv").suffix == ".csv"
