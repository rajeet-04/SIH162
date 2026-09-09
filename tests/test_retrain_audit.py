import zipfile

import numpy as np
import pandas as pd


def test_audit_deduplicates_without_deleting_and_excludes_bad_zip(tmp_path):
    from thermis.retrain_audit import audit_sources
    src = tmp_path / 'raw'
    src.mkdir()
    (src / 'one.csv').write_text('latitude,longitude,time\n20,80,2026-01-01\n')
    (src / 'two.csv').write_bytes((src / 'one.csv').read_bytes())
    (src / 'bad.zip').write_bytes(b'incomplete')
    result = audit_sources({'structured': str(src)}, tmp_path / 'audit')
    files = pd.read_parquet(result['manifest'])
    assert files.sha256.notna().all()
    assert (files.status == 'duplicate').sum() == 1
    assert files.loc[files.path.str.endswith('bad.zip'), 'status'].item() == 'unreadable'
    assert len(list(src.iterdir())) == 3


def test_npz_requires_semantics_not_just_shapes(tmp_path):
    from thermis.retrain_audit import inspect_asset
    path = tmp_path / 'patch.npz'
    np.savez(path, image=np.ones((12, 8, 8)), label=np.zeros((8, 8)))
    row = inspect_asset(path, 'satellite_patches')
    assert row['training_admitted'] is False
    assert 'provenance' in row['exclusion_reason']


def test_archive_pairing_checks_actual_member_names(tmp_path):
    from thermis.retrain_audit import archive_pairing
    images, masks = tmp_path / 'images.zip', tmp_path / 'masks.zip'
    with zipfile.ZipFile(images, 'w') as z:
        z.writestr('images/a.tiff', 'a')
    with zipfile.ZipFile(masks, 'w') as z:
        z.writestr('masks/b.tiff', 'b')
    result = archive_pairing(images, masks)
    assert result['paired'] == 0
    assert result['missing_images'] == 1
