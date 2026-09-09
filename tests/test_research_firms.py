from datetime import date

import httpx
import pytest


def test_preflight_excludes_rejected_keys_and_never_logs_them():
    from thermis.research_firms import validate_keys

    def respond(request):
        return httpx.Response(
            401 if "/bad/" in request.url.path else 200,
            text="data_id,min_date,max_date\nVIIRS_SNPP_SP,2012-01-20,2026-04-27",
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        assert validate_keys(["good", "bad"], http, interval=0) == ["good"]


def test_rotation_is_bounded_and_rate_limit_stops_all_keys(tmp_path):
    from thermis.research_firms import FirmsArchiveClient

    seen = []

    def respond(request):
        seen.append(request.url.path)
        return httpx.Response(429)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        client = FirmsArchiveClient(["secret-one", "secret-two"], http, interval=0)
        with pytest.raises(RuntimeError, match="rate limit") as error:
            client.fetch("VIIRS_SNPP_SP", date(2025, 1, 1), 5)
        assert "secret" not in str(error.value)
        with pytest.raises(RuntimeError, match="stopped"):
            client.fetch("VIIRS_SNPP_SP", date(2025, 1, 6), 5)
    assert len(seen) == 1


def test_archive_preserves_raw_types_and_resumes_without_requests(tmp_path):
    import pandas as pd

    from thermis.research_firms import FirmsArchiveClient, download_archive

    csv = (
        "latitude,longitude,acq_date,acq_time,frp,bright_ti4,type\n20,80,2025-01-01,0345,12,320,2\n"
    )
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=csv))
    ) as http:
        client = FirmsArchiveClient(["one", "two"], http, interval=0)
        report = download_archive(client, tmp_path, date(2025, 1, 1), date(2025, 1, 1))
    assert report["rows"] == 1
    frame = pd.read_parquet(tmp_path / "observations.parquet")
    assert frame.firms_type.tolist() == ["2"]
    assert "target" not in frame
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: pytest.fail("unneeded request"))
    ) as http:
        report = download_archive(
            FirmsArchiveClient(["one"], http, interval=0),
            tmp_path,
            date(2025, 1, 1),
            date(2025, 1, 1),
        )
    assert report["reused_chunks"] == 1


def test_archive_rejects_out_of_window_response(tmp_path):
    from thermis.research_firms import FirmsArchiveClient, download_archive

    csv = "latitude,longitude,acq_date,acq_time,frp,bright_ti4\n20,80,2025-02-01,0345,12,320\n"
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=csv))
    ) as http:
        with pytest.raises(ValueError, match="requested"):
            download_archive(
                FirmsArchiveClient(["one"], http, interval=0),
                tmp_path,
                date(2025, 1, 1),
                date(2025, 1, 1),
            )
    assert not (tmp_path / "observations.parquet").exists()
