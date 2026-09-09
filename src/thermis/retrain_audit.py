"""Read-only source inspection with versioned, checksummed audit outputs."""

import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_asset(path: Path, family: str) -> dict:
    row = dict(path=str(path.resolve()), family=family, bytes=path.stat().st_size,
               sha256=None, status='readable', training_admitted=False,
               exclusion_reason='requires_event_alignment_and_label_provenance',
               acquisition_date=None, spatial_coverage=None, metadata=None)
    date = re.search(r'(20\d{6})(?:\D|$)', path.stem)
    if date:
        try:
            row['acquisition_date'] = datetime.strptime(date[1], '%Y%m%d').date().isoformat()
        except ValueError:
            pass
    try:
        row['sha256'] = checksum(path)
        suffix = path.suffix.lower()
        if suffix == '.zip':
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                # Full CRC is deliberately not implied by central-directory readability.
                row['metadata'] = json.dumps({'members': len(names), 'crc_verified': False})
        elif suffix == '.csv':
            frame = pd.read_csv(path)
            row['metadata'] = json.dumps({'rows': len(frame), 'columns': list(frame.columns),
                                          'duplicate_rows': int(frame.duplicated().sum())})
            if {'latitude', 'longitude'}.issubset(frame):
                row['spatial_coverage'] = json.dumps({c: [float(frame[c].min()),
                    float(frame[c].max())] for c in ('latitude', 'longitude')})
            time_col = next((c for c in ('timestamp_utc', 'acq_date') if c in frame), None)
            if time_col:
                row['acquisition_date'] = str(frame[time_col].min())
        elif suffix == '.npz':
            with np.load(path, allow_pickle=False) as patch:
                row['metadata'] = json.dumps({k: {'shape': patch[k].shape,
                    'dtype': str(patch[k].dtype)} for k in patch.files})
            row['exclusion_reason'] = 'missing_band_scaling_georeference_and_label_provenance'
        elif suffix == '.nc4':
            import xarray as xr
            with xr.open_dataset(path, decode_times=False) as ds:
                row['metadata'] = json.dumps({'dimensions': dict(ds.sizes),
                    'variables': {k: v.attrs.get('units') for k, v in ds.data_vars.items()}})
                if 'lat' in ds and 'lon' in ds:
                    row['spatial_coverage'] = json.dumps({c: [float(ds[c].min()),
                        float(ds[c].max())] for c in ('lat', 'lon')})
            row['exclusion_reason'] = 'environmental_flux_deferred_not_fire_class_labels'
        elif suffix == '.hdf':
            with path.open('rb') as handle:
                if handle.read(4) != b'\x0e\x03\x13\x01':
                    raise ValueError('Not HDF4')
            row['metadata'] = json.dumps({'header': 'HDF4', 'raster_values_verified': False})
            row['exclusion_reason'] = 'requires_HDF4_reader_and_retrospective_label_join'
        elif suffix == '.part':
            row['status'], row['exclusion_reason'] = 'incomplete', 'partial_download'
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError) as error:
        row['status'] = 'unreadable'
        row['exclusion_reason'] = type(error).__name__
    return row


def archive_pairing(images: Path, masks: Path) -> dict:
    def names(path):
        with zipfile.ZipFile(path) as archive:
            return {Path(n).stem.replace('_patch', '').replace('_mask', '')
                    for n in archive.namelist() if n.lower().endswith(('.tif', '.tiff'))}
    image_names, mask_names = names(images), names(masks)
    return dict(paired=len(image_names & mask_names),
                missing_images=len(mask_names - image_names),
                missing_masks=len(image_names - mask_names))


def audit_sources(sources: dict[str, str], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    rows, seen = [], {}
    for family, root in sources.items():
        root = Path(root)
        if not root.exists():
            rows.append(dict(path=str(root), family=family, status='missing', sha256=None,
                             training_admitted=False, exclusion_reason='source_missing'))
            continue
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in (
                '.csv', '.hdf', '.nc4', '.npz', '.zip', '.tif', '.tiff', '.part'
            ) or any(p.startswith('.') for p in path.relative_to(root).parts):
                continue
            row = inspect_asset(path, family)
            digest = row['sha256']
            if digest and row['status'] == 'readable':
                if digest in seen:
                    row.update(status='duplicate', duplicate_of=seen[digest],
                               exclusion_reason='byte_identical_duplicate')
                else:
                    seen[digest] = row['path']
            rows.append(row)
            if len(rows) % 250 == 0:
                print(f'Audited {len(rows)} assets', flush=True)
    frame = pd.DataFrame(rows)
    manifest = output / 'assets.parquet'
    frame.to_parquet(manifest, index=False)
    report = dict(version=datetime.now(timezone.utc).isoformat(), manifest=str(manifest),
                  assets=len(frame), status_counts=frame.status.value_counts().to_dict(),
                  originals_modified=False, manifest_sha256=checksum(manifest))
    (output / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
