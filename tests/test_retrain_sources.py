import pytest


def test_osm_recovery_reuses_snapshots_and_fetches_only_missing(tmp_path, monkeypatch):
    import json

    import httpx
    import pandas as pd

    from thermis.retrain_sources import fetch_historical_facilities

    prior = tmp_path / 'prior'
    prior.mkdir()
    stamp = '2026-06-01T00:00:00Z'
    (prior / 'report.json').write_text(json.dumps({'as_of': stamp}))
    for name in ('jamnagar', 'mumbai', 'bokaro', 'paradip'):
        (prior / f'{name}.json').write_text(json.dumps({'elements': [
            {'type': 'node', 'id': 1, 'lat': 20., 'lon': 80., 'tags': {}}
        ]}))
    def post(self, url, data):
        # Re-requesting any successful area must fail this test.
        assert '20.7,72.3,21.7,73.5' in data['data'] or '20.7,80.9,22.0,82.2' in data['data']
        return httpx.Response(200, request=httpx.Request('POST', url), json={'elements': [
            {'type': 'node', 'id': 2, 'lat': 21., 'lon': 81., 'tags': {}}
        ]})
    monkeypatch.setattr(httpx.Client, 'post', post)
    report = fetch_historical_facilities(tmp_path / 'next', stamp, reuse_from=prior)
    assert report['features'] == 2
    assert report['areas']['jamnagar']['reused'] is True
    assert report['areas']['surat']['reused'] is False
    assert len(pd.read_parquet(tmp_path / 'next' / 'facilities.parquet')) == 2
    assert (prior / 'jamnagar.json').exists()


def test_osm_recovery_refuses_different_snapshot_before_creating_output(tmp_path):
    import json

    from thermis.retrain_sources import fetch_historical_facilities
    prior = tmp_path / 'prior'
    prior.mkdir()
    (prior / 'report.json').write_text(json.dumps({'as_of': '2025-01-01T00:00:00Z'}))
    with pytest.raises(ValueError, match='snapshot'):
        fetch_historical_facilities(tmp_path / 'next', reuse_from=prior)
    assert not (tmp_path / 'next').exists()


def test_osm_snapshot_keeps_distinct_flare_and_facility_types():
    from thermis.retrain_sources import osm_points
    points = osm_points({'elements': [
        {'type': 'node', 'id': 1, 'lat': 20., 'lon': 80., 'tags': {'industrial': 'steel'}},
        {'type': 'node', 'id': 2, 'lat': 21., 'lon': 81., 'tags': {'man_made': 'flare'}},
    ]}, '2026-06-01T00:00:00Z')
    assert points.kind.tolist() == ['industrial', 'flare']
    assert points.available_from.tolist() == ['2026-06-01T00:00:00Z'] * 2


def test_download_refuses_untrusted_origin_before_request(tmp_path):
    from thermis.retrain_sources import recover_archive
    with pytest.raises(ValueError, match='origin'):
        recover_archive('https://example.com/file.zip', tmp_path/'file.zip', '0'*32)


def test_download_does_not_overwrite_existing_archive(tmp_path):
    from thermis.retrain_sources import recover_archive
    dest = tmp_path/'fire.zip'
    dest.write_bytes(b'original')
    with pytest.raises(FileExistsError):
        recover_archive('https://zenodo.org/records/17619196/files/fire_patches.zip',
                        dest, '0'*32)
    assert dest.read_bytes() == b'original'
