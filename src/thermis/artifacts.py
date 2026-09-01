from pathlib import Path
from typing import Any

import joblib


def save_bundle(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> dict[str, Any]:
    value = joblib.load(path)
    if not isinstance(value, dict):
        raise ValueError("model bundle must be a dictionary")
    return value
