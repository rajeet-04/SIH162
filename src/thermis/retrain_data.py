"""Leakage-aware event preparation. Proxy evidence is never reference ground truth."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

EARTH_KM = 6371.0088
MODEL_FEATURES = (
    'frp', 'brightness_temperature', 'prior_detections_7d', 'prior_detections_30d',
    'prior_detections_90d', 'prior_active_days_90d', 'prior_frp_median_30d',
    'frp_ratio_30d', 'nearest_flare_distance_m', 'nearest_industrial_distance_m',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'history_days_available',
    'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'precipitation',
)


def read_observations(database: Path) -> pd.DataFrame:
    """Read a live SQLite snapshot without updates or migrations."""
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as db:
        rows = [json.loads(row[0]) for row in db.execute('SELECT observation FROM events')]
    return canonicalize(pd.DataFrame(rows))


def canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result['timestamp_utc'] = pd.to_datetime(result.timestamp_utc, utc=True, errors='coerce')
    for column in ('latitude', 'longitude', 'frp', 'brightness_temperature'):
        result[column] = pd.to_numeric(result[column], errors='coerce')
    valid = (result.latitude.between(-90, 90) & result.longitude.between(-180, 180)
             & result.frp.ge(0) & result.timestamp_utc.notna())
    result = result.loc[valid].copy()
    # This is an India-region bounding box, NOT a political-boundary country filter.
    result = result[result.latitude.between(4, 38) & result.longitude.between(68, 98)]
    keys = ['timestamp_utc', 'latitude', 'longitude']
    keys += [c for c in ('product', 'satellite') if c in result]
    result = result.drop_duplicates(keys).drop_duplicates('event_id')
    return result.sort_values(['timestamp_utc', 'event_id']).reset_index(drop=True)


def temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    result['timestamp_utc'] = pd.to_datetime(result.timestamp_utc, utc=True)
    coords = np.radians(result[['latitude', 'longitude']].to_numpy(float))
    tree = BallTree(coords, metric='haversine')
    seconds = result.timestamp_utc.astype('int64').to_numpy() / 1e9
    frp = result.frp.to_numpy(float)
    values = {f'prior_detections_{d}d': np.zeros(len(result)) for d in (7, 30, 90)}
    values.update(prior_active_days_90d=np.zeros(len(result)),
                  prior_frp_median_30d=np.full(len(result), np.nan))
    for index in range(len(result)):
        nearby = tree.query_radius(coords[index:index + 1], r=5 / EARTH_KM)[0]
        age = seconds[index] - seconds[nearby]
        for days in (7, 30, 90):
            selected = nearby[(age > 0) & (age <= days * 86400)]
            values[f'prior_detections_{days}d'][index] = len(selected)
            if days == 90:
                values['prior_active_days_90d'][index] = len(np.unique(
                    np.floor(seconds[selected] / 86400)))
            if days == 30 and len(selected):
                valid_frp = frp[selected][np.isfinite(frp[selected])]
                if len(valid_frp):
                    values['prior_frp_median_30d'][index] = np.median(valid_frp)
    for key, value in values.items():
        result[key] = value
    result['frp_ratio_30d'] = result.frp / result.prior_frp_median_30d.clip(lower=0.1)
    result['history_days_available'] = ((result.timestamp_utc - result.timestamp_utc.min())
                                        .dt.total_seconds() / 86400).clip(upper=90)
    for unit, period in (('hour', 24), ('month', 12)):
        angle = getattr(result.timestamp_utc.dt, unit) * 2 * np.pi / period
        result[f'{unit}_sin'], result[f'{unit}_cos'] = np.sin(angle), np.cos(angle)
    result['feature_as_of_utc'] = result.timestamp_utc
    return result


def join_points(frame: pd.DataFrame, points: pd.DataFrame, kind: str) -> pd.DataFrame:
    result = frame.copy()
    column = f'nearest_{kind}_distance_m'
    result[column] = np.nan
    if points.empty:
        return result
    # Unknown snapshot dates cannot establish historical availability.
    if 'available_from' not in points:
        return result
    points = points.dropna(subset=['latitude', 'longitude', 'available_from']).copy()
    points['available_from'] = pd.to_datetime(points.available_from, utc=True)
    stamps = pd.to_datetime(result.timestamp_utc, utc=True)
    for available, group in points.groupby('available_from'):
        eligible = stamps >= available
        if not eligible.any():
            continue
        tree = BallTree(np.radians(group[['latitude', 'longitude']].to_numpy(float)),
                        metric='haversine')
        distance, _ = tree.query(np.radians(
            result.loc[eligible, ['latitude', 'longitude']].to_numpy(float)), k=1)
        current = result.loc[eligible, column].to_numpy(float)
        result.loc[eligible, column] = np.fmin(current, distance[:, 0] * EARTH_KM * 1000)
    return result


def label_examples(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    targets, tiers, sources = [], [], []
    for row in result.to_dict('records'):
        evidence = []
        flare = row.get('nearest_flare_distance_m', np.nan)
        facility = row.get('nearest_industrial_distance_m', np.nan)
        days = row.get('prior_active_days_90d', 0)
        ratio = row.get('frp_ratio_30d', np.nan)
        if flare <= 1000 and days >= 5 and ratio <= 2:
            evidence.append(('persistent_industrial_heat_or_flare', 'flare_catalog_and_history'))
        if facility <= 1500 and days >= 5 and ratio >= 3:
            evidence.append(('industrial_fire_candidate', 'facility_and_intensity_change'))
        burned = row.get('burned_overlap') is True
        if burned and row.get('landcover') in ('forest', 'grassland', 'shrubland'):
            evidence.append(('vegetation_fire_candidate', 'burned_area_and_landcover'))
        if burned and row.get('landcover') == 'cropland':
            evidence.append(('agricultural_burn_candidate', 'burned_area_and_cropland'))
        if len(evidence) == 1:
            target, source = evidence[0]
            tier = 'weak'
        else:
            target, tier = 'uncertain', 'unknown'
            source = 'conflicting_evidence' if evidence else 'insufficient_evidence'
        targets.append(target)
        tiers.append(tier)
        sources.append(source)
    result['target'], result['label_tier'], result['label_provenance'] = targets, tiers, sources
    return result


def group_events(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(['timestamp_utc', 'event_id']).reset_index(drop=True).copy()
    groups = DBSCAN(eps=1 / EARTH_KM, min_samples=1, metric='haversine',
                    algorithm='ball_tree').fit_predict(
        np.radians(result[['latitude', 'longitude']].to_numpy(float)))
    result['site_id'] = [f'site-{g}' for g in groups]
    result['episode_id'] = ''
    for site, group in result.groupby('site_id'):
        gaps = pd.to_datetime(group.timestamp_utc, utc=True).diff().dt.total_seconds()
        episode = (gaps.isna() | gaps.gt(48 * 3600)).cumsum()
        result.loc[group.index, 'episode_id'] = [f'{site}-{n}' for n in episode]
    result['region_id'] = (np.floor(result.latitude / 2).astype(int).astype(str) + ':'
                           + np.floor(result.longitude / 2).astype(int).astype(str))
    return result


def chronological_split(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    times = pd.to_datetime(result.timestamp_utc, utc=True)
    order = times.sort_values()
    if len(order) < 10:
        raise ValueError('At least ten observations are required for chronological splitting')
    # 80% fit, 10% validation, newest 10% untouched. Equal acquisition times stay together.
    validation_start = order.iloc[int(len(order) * .8)]
    test_start = order.iloc[int(len(order) * .9)]
    result['split'] = np.where(times >= test_start, 'test',
                               np.where(times >= validation_start, 'validation', 'train'))
    overlap = result.groupby('episode_id').split.nunique()
    result.loc[result.episode_id.isin(overlap[overlap > 1].index), 'split'] = 'purged'
    return result


def prepare_examples(events: pd.DataFrame, output: Path,
                     flare_points: pd.DataFrame | None = None,
                     industrial_points: pd.DataFrame | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    frame = canonicalize(events)
    if frame.empty:
        raise ValueError('No valid India-region observations')
    frame = temporal_features(frame)
    for kind, points in (('flare', flare_points), ('industrial', industrial_points)):
        frame = join_points(frame, points if points is not None else pd.DataFrame(), kind)
    frame = chronological_split(group_events(label_examples(frame)))
    for column in MODEL_FEATURES:
        if column not in frame:
            frame[column] = np.nan
    frame.to_parquet(output / 'examples.parquet', index=False)
    report = {
        'rows': len(frame), 'split_rows': frame.split.value_counts().to_dict(),
        'label_counts': frame.groupby(['split', 'label_tier', 'target']).size().to_dict(),
        'feature_columns': list(MODEL_FEATURES), 'reference_labels': 0,
        'scope': 'India-region bbox 68E,4N,98E,38N; not an India boundary mask',
        'missing_features': [c for c in MODEL_FEATURES if frame[c].isna().all()],
        'promotion_blocked': True,
        'reason': 'Proxy labels only; industrial accidents not independently verified',
    }
    report['label_counts'] = {'|'.join(k): int(v) for k, v in report['label_counts'].items()}
    (output / 'preparation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
