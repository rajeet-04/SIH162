from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, recall_score

from thermis.artifacts import save_bundle
from thermis.features import FEATURE_COLUMNS


def persistent_false_alert_rate(truth: np.ndarray, pred: np.ndarray) -> float:
    persistent = truth == "persistent"
    return float(np.mean(pred[persistent] != "persistent")) if np.any(persistent) else 0.0


def weighted_ranking_score(metrics: dict[str, float]) -> float:
    weights = {
        "industrial_recall": 0.30,
        "macro_f1": 0.20,
        "persistent_specificity": 0.15,
        "pr_auc": 0.15,
        "calibration_quality": 0.10,
        "speed_quality": 0.10,
    }
    return sum(weights[key] * metrics[key] for key in weights)


def _fit_one(
    frame: pd.DataFrame, target: str, feature_columns: list[str], seed: int, output: Path
) -> dict[str, Any]:
    development = frame[frame["split"] != "ranking"].copy()
    if development.empty:
        raise ValueError("training data cannot be empty")
    if (frame["split"] == "ranking").any() and development.index.intersection(
        frame.index[frame["split"] == "ranking"]
    ).any():
        raise ValueError("ranking rows crossed into training")
    development = development[development["supervised_eligible"]].dropna(subset=[target])
    if development[target].nunique() < 2:
        raise ValueError(f"{target} needs at least two eligible classes")
    model = CatBoostClassifier(
        iterations=600,
        depth=7,
        learning_rate=0.05,
        loss_function="MultiClass",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(development[feature_columns], development[target])
    pred = model.predict(development[feature_columns]).ravel()
    metrics = {
        "macro_f1": float(f1_score(development[target], pred, average="macro")),
        "recall": float(recall_score(development[target], pred, average="macro")),
    }
    bundle = {
        "model": model,
        "target": target,
        "feature_columns": feature_columns,
        "class_order": [str(value) for value in model.classes_],
        "metrics": metrics,
        "training_rows": int(len(development)),
        "ranking_rows_used": 0,
    }
    save_bundle(bundle, output)
    return bundle


def train_tabular(
    frame: pd.DataFrame,
    output: Path,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    seed: int = 26162,
) -> dict[str, dict[str, Any]]:
    available = [column for column in feature_columns if column in frame.columns]
    if not available:
        raise ValueError("no configured feature columns are present")
    output.mkdir(parents=True, exist_ok=True)
    return {
        "stage1": _fit_one(frame, "label_stage_1", available, seed, output / "stage1.joblib"),
        "stage2": _fit_one(frame, "label_stage_2", available, seed, output / "stage2.joblib"),
    }
