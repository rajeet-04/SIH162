from pathlib import Path

from thermis.artifacts import load_bundle, save_bundle


def test_model_bundle_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "model.joblib"
    save_bundle({"class_order": ["a", "b"], "training_rows": 4}, path)
    assert load_bundle(path)["training_rows"] == 4
