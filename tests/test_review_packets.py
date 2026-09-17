import pandas as pd
import pytest

from thermis.review_packets import build_packets, validate_packets


def observations():
    return pd.DataFrame(
        [
            dict(
                event_id=f"e{i}",
                timestamp_utc="2024-01-03T06:00:00Z",
                latitude=10 + i,
                longitude=75,
                frp=i + 1,
                brightness_temperature=330,
            )
            for i in range(12)
        ]
    )


def test_packets_have_exclusive_ownership_and_no_sealed_rows(tmp_path):
    frame = observations()
    future = frame.iloc[:1].assign(event_id="sealed", timestamp_utc="2025-06-01T00:00:00Z")
    build_packets(
        pd.concat([frame, frame.iloc[:1], future]),
        tmp_path / "review",
        ["alice", "bob"],
        count=12,
        cutoff="2025-05-18T08:12:00Z",
    )
    a = pd.read_csv(tmp_path / "review/alice.csv")
    b = pd.read_csv(tmp_path / "review/bob.csv")
    assert len(a) == len(b) == 6
    assert set(a.review_id).isdisjoint(b.review_id)
    assert set(a.cell_id).isdisjoint(b.cell_id)
    assert set(a.owner) == {"alice"}
    assert set(b.owner) == {"bob"}
    assert a.label.isna().all()
    assert a.observation_count.eq(1).all()
    assert "sealed" not in set(a.representative_event_id) | set(b.representative_event_id)
    assert validate_packets(tmp_path / "review")["pending"] == 12


def test_worldview_urls_open_with_reviewer_ready_layers_and_viewport(tmp_path):
    root = tmp_path / "review"
    build_packets(observations(), root, ["alice"], count=6, cutoff="2025-01-01")
    packet = pd.read_csv(root / "alice.csv")
    row = packet.iloc[0]
    lat = float(row.latitude)
    lon = float(row.longitude)
    expected_view = f"v={lon - 0.04:.5f},{lat - 0.02:.5f},{lon + 0.04:.5f},{lat + 0.02:.5f}"
    url = row.worldview_url
    assert expected_view in url
    assert "VIIRS_SNPP_Thermal_Anomalies_375m_All" in url
    assert "VIIRS_SNPP_CorrectedReflectance_TrueColor" in url
    assert "VIIRS_NOAA21_CorrectedReflectance_TrueColor(hidden)" in url
    assert "VIIRS_NOAA20_CorrectedReflectance_TrueColor(hidden)" in url
    assert "OCI_PACE_True_Color(hidden)" in url
    assert "Reference_Labels_15m(hidden)" in url
    assert "DoS_International_Boundaries(hidden)" in url
    assert "Coastlines_15m" in url
    assert "&lg=true" in url
    assert "&t=2024-01-03-T00%3A00%3A00Z" in url


def test_validation_rejects_changed_owner_and_unsupported_label(tmp_path):
    root = tmp_path / "review"
    build_packets(observations(), root, ["alice"], count=6, cutoff="2025-01-01")
    frame = pd.read_csv(root / "alice.csv", keep_default_na=False)
    frame.loc[0, "owner"] = "bob"
    frame.to_csv(root / "alice.csv", index=False)
    with pytest.raises(ValueError, match="immutable"):
        validate_packets(root)
    frame.loc[0, "owner"] = "alice"
    frame.loc[0, "review_status"] = "reviewed"
    frame.loc[0, "label"] = "industrial_fire_candidate"
    frame.to_csv(root / "alice.csv", index=False)
    with pytest.raises(ValueError, match="evidence"):
        validate_packets(root)
    frame.loc[0, "label"] = "uncertain"
    frame.loc[0, "notes"] = "Clouds obscure the observation window"
    frame.to_csv(root / "alice.csv", index=False)
    assert validate_packets(root)["reviewed"] == 1


def test_refuses_overwrite_and_duplicate_reviewers(tmp_path):
    with pytest.raises(ValueError):
        build_packets(observations(), tmp_path / "bad", ["alice", "alice"], 6, "2025-01-01")
    root = tmp_path / "review"
    build_packets(observations(), root, ["alice"], 6, "2025-01-01")
    with pytest.raises(FileExistsError):
        build_packets(observations(), root, ["alice"], 6, "2025-01-01")
