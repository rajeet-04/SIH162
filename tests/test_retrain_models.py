"""Synthetic-only contracts for isolated challenger training."""

import importlib
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def api():
    class API:
        def __getattr__(self, name):
            assert importlib.util.find_spec("thermis.retrain_models") is not None, (
                "Task 2 challenger implementation is missing"
            )
            return getattr(importlib.import_module("thermis.retrain_models"), name)

    return API()


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2020-01-01", periods=20, tz="UTC"),
            "event_id": [f"e{i // 2}" for i in range(20)],
            "episode_id": [f"p{i // 2}" for i in range(20)],
            "site_id": ["a"] * 16 + ["new"] * 4,
            "region_id": ["r"] * 16 + ["new"] * 4,
            "split": ["train"] * 12 + ["validation"] * 6 + ["test"] * 2,
            "target": ["background", "industrial"] * 10,
            "label_tier": ["reference"] * 8 + ["weak"] * 12,
            "heat": [0.0, 3.0] * 10,
            "context": [np.nan, 1.0, 2.0, 3.0] * 5,
        }
    )


def test_insufficient_classes_fails_closed(api, frame, tmp_path):
    frame.loc[frame.split == "train", "target"] = "industrial"
    report = api.train_challengers(frame, tmp_path, ["heat"], device="cpu")
    assert report["status"] == "blocked"
    assert "two" in report["reason"]
    assert not list(tmp_path.glob("*.pt"))
    json.dumps(report, allow_nan=False)


def test_train_only_preprocessing_and_strict_lazy_history(api, frame):
    frame.loc[1, "split"] = "test"  # even an early sealed row cannot be history
    frame.loc[2, "timestamp_utc"] = frame.loc[0, "timestamp_utc"]
    frame.loc[3, "label_tier"] = "unknown"
    frame.loc[3, "heat"] = 100.0
    frame.loc[19, "heat"] = 1e12
    prep = api.Preprocessor.fit(frame, ["heat", "context", "absent"])
    assert prep.medians[0] < 100
    assert prep.medians[2] == 0
    transformed = prep.transform(frame)
    assert np.isfinite(transformed).all()
    assert transformed[0, 4] == 1  # context NaN flag
    assert (transformed[:, 5] == 1).all()
    dataset = api.SequenceDataset(frame, prep, ["background", "industrial"], "train")
    assert 1 not in dataset.rows and 3 not in dataset.rows
    current = dataset.rows.index(4)
    assert dataset.history_indices(4) == [2, 3]
    sequence, numeric, length, target = dataset[current]
    assert sequence.shape == (16, 6) and numeric.shape == (6,)
    assert length == 2 and target == 0
    assert dataset.history_indices(0) == []
    assert dataset.history_indices(2) == []  # ties excluded
    assert all(frame.loc[i, "site_id"] == "new" for i in dataset.history_indices(17))


def test_real_cpu_roundtrip_and_sealed_invariance(api, frame, tmp_path):
    first = api.train_challengers(
        frame, tmp_path / "one", ["heat", "context", "absent"], epochs=1, batch_size=4, device="cpu"
    )
    altered = frame.copy()
    altered.loc[altered.split == "test", "heat"] = 1e12
    altered.loc[altered.split == "test", "target"] = "sealed secret"
    second = api.train_challengers(
        altered,
        tmp_path / "two",
        ["heat", "context", "absent"],
        epochs=1,
        batch_size=4,
        device="cpu",
    )
    assert first["status"] == second["status"] == "complete"
    assert first["identity"] == second["identity"]
    assert first["promotion"]["allowed"] is False
    assert first["validation_rows"] == 6
    for name in ("catboost", "gru"):
        left = api.load_challenger(first["artifacts"][name], device="cpu")
        right = api.load_challenger(second["artifacts"][name], device="cpu")
        assert list(left.classes_) == ["background", "industrial"]
        probs = left.predict_proba(frame)
        np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(probs, right.predict_proba(frame), atol=1e-7)
        assert np.isfinite(left.predict_proba(frame.drop(columns="context"))).all()
        assert first["metrics"][name] == second["metrics"][name]
        assert first["metrics"][name]["reference"]["macro_f1"] is None
        assert first["metrics"][name]["unseen_site"]["rows"] == 2
    checkpoint = torch.load(
        first["artifacts"]["checkpoint"], map_location="cpu", weights_only=False
    )
    assert {"model", "optimizer", "scaler", "rng", "identity", "config"} <= checkpoint.keys()
    json.dumps(first, allow_nan=False)
    assert (tmp_path / "one" / "model_card.md").exists()


def test_unknown_labels_never_fit_or_score(api, frame, tmp_path):
    frame.loc[[0, 12], "label_tier"] = "unknown"
    frame.loc[[0, 12], "target"] = "secret"
    frame.loc[1, "target"] = None
    report = api.train_challengers(frame, tmp_path, ["heat"], epochs=1, device="cpu")
    assert report["training_rows"] == 10
    assert report["validation_rows"] == 5
    assert report["classes"] == ["background", "industrial"]


def test_metrics_unsupported_and_persistent_events(api, frame):
    empty = api.evaluate_metrics(frame.iloc[:0], np.empty((0, 2)), ["background", "industrial"])
    assert empty["macro_f1"] is None and empty["ece"] is None
    single = frame.iloc[:2].copy()
    single["target"] = "background"
    metrics = api.evaluate_metrics(
        single, np.array([[0.1, 0.9], [0.2, 0.8]]), ["background", "industrial"]
    )
    assert metrics["per_class"]["industrial"]["pr_auc"] is None
    assert metrics["per_class"]["industrial"]["recall"] is None
    assert metrics["persistent_false_industrial_alerts"]["events"] == 1
    assert metrics["persistent_false_industrial_alerts"]["rate"] == 1


def test_resume_rejects_changed_data_and_config(api, frame, tmp_path):
    api.train_challengers(frame, tmp_path, ["heat"], epochs=1, device="cpu")
    changed = frame.copy()
    changed.loc[0, "heat"] = 42
    with pytest.raises(ValueError, match="identity"):
        api.train_challengers(changed, tmp_path, ["heat"], epochs=2, device="cpu", resume=True)
    with pytest.raises(ValueError, match="identity"):
        api.train_challengers(
            frame, tmp_path, ["heat"], epochs=2, batch_size=8, device="cpu", resume=True
        )


def test_resume_matches_uninterrupted_training(api, frame, tmp_path):
    api.train_challengers(frame, tmp_path / "resume", ["heat"], epochs=1, device="cpu")
    resumed = api.train_challengers(
        frame, tmp_path / "resume", ["heat"], epochs=2, device="cpu", resume=True
    )
    direct = api.train_challengers(frame, tmp_path / "direct", ["heat"], epochs=2, device="cpu")
    a = api.load_challenger(resumed["artifacts"]["gru"])
    b = api.load_challenger(direct["artifacts"]["gru"])
    np.testing.assert_allclose(a.predict_proba(frame), b.predict_proba(frame), atol=1e-7)


def test_oom_restarts_epoch_and_is_bounded(api):
    # Inject allocator failure after an optimizer-like mutation: retry must undo it.
    state = {"value": 0}
    batches = []

    def run(batch):
        batches.append(batch)
        state["value"] += 1
        if batch > 2:
            raise torch.OutOfMemoryError("synthetic allocator failure")

    def restore():
        state["value"] = 0

    assert api.run_epoch_with_backoff(run, restore, 8) == 2
    assert state["value"] == 1 and batches == [8, 4, 2]
    with pytest.raises(torch.OutOfMemoryError):
        api.run_epoch_with_backoff(
            lambda _: (_ for _ in ()).throw(torch.OutOfMemoryError()), restore, 1
        )


def test_refuses_deployed_output(api, frame, tmp_path):
    with pytest.raises(ValueError, match="deployed"):
        api.train_challengers(frame, tmp_path / "models" / "tabular", ["heat"], device="cpu")


def test_history_is_bounded_and_ignores_future_and_purged(api, frame):
    extra = pd.concat([frame.iloc[:1]] * 40, ignore_index=True)
    extra["timestamp_utc"] = pd.date_range("2021-01-01", periods=40, tz="UTC")
    extra["split"] = "train"
    extra.loc[38, "split"] = "purged"
    dataset = api.SequenceDataset(
        extra, api.Preprocessor.fit(frame, ["heat"]), ["background", "industrial"], "train"
    )
    assert dataset.history_indices(39) == list(range(22, 38))
    assert dataset.history_indices(3) == [0, 1, 2]


def test_oom_after_optimizer_update_restores_full_training(api, frame, tmp_path, monkeypatch):
    original = torch.optim.AdamW.step
    failures = [True, True]

    def fail_after_update(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if self.state[next(iter(self.state))]["step"].item() >= 2 and failures:
            failures.pop()
            raise torch.OutOfMemoryError("synthetic allocation after real AdamW update")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, "step", fail_after_update)
        recovered = api.train_challengers(
            frame, tmp_path / "retry", ["heat"], epochs=2, batch_size=8, device="cpu"
        )
    direct = api.train_challengers(
        frame, tmp_path / "clean", ["heat"], epochs=2, batch_size=2, device="cpu"
    )
    assert recovered["active_batch_size"] == 2
    a = torch.load(recovered["artifacts"]["checkpoint"], weights_only=False)
    b = torch.load(direct["artifacts"]["checkpoint"], weights_only=False)
    for name, tensor in a["model"].items():
        # Microbatch regrouping changes float32 summation at sub-micro precision.
        torch.testing.assert_close(tensor, b["model"][name], atol=1e-6, rtol=1e-5)
    assert a["optimizer"]["state"][0]["step"].item() == 2


def test_cpu_checkpoint_does_not_initialize_cuda_rng(api, frame, tmp_path, monkeypatch):
    def forbidden():
        pytest.fail("CPU training must not initialize CUDA RNG/context")

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden)
    api.train_challengers(frame, tmp_path, ["heat"], epochs=1, device="cpu")


def test_validation_cannot_change_training_checkpoint(api, frame, tmp_path):
    first = api.train_challengers(frame, tmp_path / "first", ["heat"], epochs=1, device="cpu")
    frame.loc[frame.split == "validation", "heat"] = 1e10
    other = api.train_challengers(frame, tmp_path / "other", ["heat"], epochs=1, device="cpu")
    a = torch.load(first["artifacts"]["checkpoint"], weights_only=False)
    b = torch.load(other["artifacts"]["checkpoint"], weights_only=False)
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], atol=0, rtol=0)
    assert a["metadata"]["preprocessing"] == b["metadata"]["preprocessing"]
