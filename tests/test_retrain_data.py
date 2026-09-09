import pandas as pd
import pytest

from thermis import features


def observations():
    return pd.DataFrame({
        'event_id': ['a', 'b', 'c'], 'latitude': [20., 20., 20.],
        'longitude': [80., 80., 80.],
        'timestamp_utc': ['2026-01-01', '2026-01-01', '2026-01-02'],
        'frp': [10., 100., 30.],
    })


def test_same_overpass_is_not_history():
    result = features.add_persistence_features(observations())
    assert result.prior_detections_7d.tolist() == [0, 0, 2]


def test_industrial_distance_is_not_flare_distance():
    points = pd.DataFrame({'latitude': [20.], 'longitude': [80.]})
    result = features.build_event_features(observations(), {'flare_points': points})
    assert result.nearest_industrial_distance_m.isna().all()
    assert result.nearest_flare_distance_m.tolist() == [0., 0., 0.]


def test_industrial_points_are_separate():
    points = pd.DataFrame({'latitude': [21.], 'longitude': [80.]})
    result = features.build_event_features(observations(), {'industrial_points': points})
    assert result.nearest_flare_distance_m.isna().all()
    assert result.nearest_industrial_distance_m.between(110000, 112000).all()


def test_temporal_features_ignore_future_and_simultaneous():
    from thermis.retrain_data import temporal_features
    result = temporal_features(observations())
    assert result.prior_detections_7d.tolist() == [0, 0, 2]
    assert pd.isna(result.iloc[1].prior_frp_median_30d)
    assert result.iloc[2].prior_frp_median_30d == 55.


def test_split_purges_cross_boundary_episodes():
    from thermis.retrain_data import chronological_split
    frame = pd.DataFrame({'timestamp_utc': pd.date_range('2026-01-01', periods=20, tz='UTC'),
                          'episode_id': [f'e{i}' for i in range(20)]})
    frame.loc[17:18, 'episode_id'] = 'cross'
    result = chronological_split(frame)
    assert result.loc[17:18, 'split'].tolist() == ['purged', 'purged']
    assert result.loc[19, 'split'] == 'test'
    assert set(result[result.split == 'train'].episode_id).isdisjoint(
        result[result.split == 'validation'].episode_id)


def test_labels_do_not_turn_missing_context_into_negative_evidence():
    from thermis.retrain_data import label_examples
    result = label_examples(observations())
    assert set(result.target) == {'uncertain'}
    assert set(result.label_tier) == {'unknown'}


def test_proxy_labels_never_become_reference():
    from thermis.retrain_data import label_examples
    frame = observations().assign(nearest_flare_distance_m=0., prior_detections_90d=20,
                                  prior_active_days_90d=10, frp_ratio_30d=1.)
    result = label_examples(frame)
    assert set(result.target) == {'persistent_industrial_heat_or_flare'}
    assert set(result.label_tier) == {'weak'}


def test_conflicting_proxy_evidence_abstains():
    from thermis.retrain_data import label_examples
    frame = observations().assign(nearest_flare_distance_m=0., prior_detections_90d=20,
                                  prior_active_days_90d=10, frp_ratio_30d=1.,
                                  burned_overlap=True, landcover='forest')
    assert set(label_examples(frame).target) == {'uncertain'}


def test_future_context_is_rejected():
    from thermis.retrain_data import join_points
    points = pd.DataFrame({'latitude': [20.], 'longitude': [80.],
                           'available_from': ['2026-01-02']})
    result = join_points(observations(), points, 'industrial')
    assert result.nearest_industrial_distance_m.iloc[:2].isna().all()
    assert result.nearest_industrial_distance_m.iloc[2] == pytest.approx(0)
