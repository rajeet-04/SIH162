import numpy as np
import pandas as pd


def sample():
    return pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(100)],
            "timestamp_utc": pd.date_range("2025-01-01", periods=100, tz="UTC"),
            "latitude": 20.0,
            "longitude": 80.0,
            "frp": np.arange(100) + 1.0,
            "brightness_temperature": 320.0,
            "daynight": "D",
        }
    )


def test_unsupervised_fit_ignores_targets_and_sealed_values(tmp_path):
    from thermis.anomaly_training import load_anomaly, train_anomaly

    a = sample()
    first = train_anomaly(a, tmp_path / "a", device="cpu", epochs=2, batch_size=32)
    a.loc[90:, "frp"] = 1e10
    a["target"] = "invented"
    second = train_anomaly(a, tmp_path / "b", device="cpu", epochs=2, batch_size=32)
    assert first["identity"] == second["identity"]
    assert first["split_rows"] == {"train": 80, "validation": 10, "test": 10}
    assert first["classification_accuracy"] is None
    left = load_anomaly(tmp_path / "a" / "anomaly.pt")
    right = load_anomaly(tmp_path / "b" / "anomaly.pt")
    np.testing.assert_allclose(left.score(sample()), right.score(sample()), atol=1e-6)
    assert np.isfinite(left.score(sample().drop(columns="brightness_temperature"))).all()


def test_duplicate_events_do_not_change_training_weight(tmp_path):
    from thermis.anomaly_training import train_anomaly

    a = sample()
    report = train_anomaly(pd.concat([a, a.iloc[:2]]), tmp_path / "run", device="cpu", epochs=1)
    assert report["rows"] == 100 and report["duplicates_excluded"] == 2


def test_invalid_measurements_are_excluded(tmp_path):
    from thermis.anomaly_training import train_anomaly

    a = sample()
    a.loc[0, "frp"] = -1
    a.loc[1, "latitude"] = 200
    report = train_anomaly(a, tmp_path / "run", device="cpu", epochs=1)
    assert report["invalid_excluded"] == 2


def test_resume_matches_uninterrupted_fit(tmp_path):
    from thermis.anomaly_training import load_anomaly, train_anomaly

    train_anomaly(sample(), tmp_path / "resume", device="cpu", epochs=1)
    train_anomaly(sample(), tmp_path / "resume", device="cpu", epochs=2, resume=True)
    train_anomaly(sample(), tmp_path / "direct", device="cpu", epochs=2)
    np.testing.assert_allclose(
        load_anomaly(tmp_path / "resume" / "anomaly.pt").score(sample()),
        load_anomaly(tmp_path / "direct" / "anomaly.pt").score(sample()),
        atol=1e-7,
    )


def test_anomaly_oom_rolls_back_optimizer_updates(tmp_path, monkeypatch):
    import torch

    from thermis.anomaly_training import load_anomaly, train_anomaly

    original = torch.optim.AdamW.step
    failed = []

    def step(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        if not failed:
            failed.append(True)
            raise torch.OutOfMemoryError("simulated after parameter update")
        return value

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, "step", step)
        report = train_anomaly(sample(), tmp_path / "retry", device="cpu", epochs=1, batch_size=32)
    train_anomaly(sample(), tmp_path / "direct", device="cpu", epochs=1, batch_size=16)
    assert report["active_batch_size"] == 16
    np.testing.assert_allclose(
        load_anomaly(tmp_path / "retry" / "anomaly.pt").score(sample()),
        load_anomaly(tmp_path / "direct" / "anomaly.pt").score(sample()),
        atol=1e-7,
    )
