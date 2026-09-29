from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
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
    if "timestamp_utc" in development:
        development = development.sort_values("timestamp_utc")
    # Three-way chronological cut: fit -> calibrate -> validate. The validation
    # slice never touches fitting or calibration; reported metrics come from it.
    n = len(development)
    fit_end = max(1, min(n - 2, int(n * 0.70)))
    cal_end = max(fit_end + 1, min(n - 1, int(n * 0.85)))
    fit_frame = development.iloc[:fit_end]
    calibration_frame = development.iloc[fit_end:cal_end]
    validation_frame = development.iloc[cal_end:]
    for part in (fit_frame, calibration_frame, validation_frame):
        if part[target].nunique() < 2:
            raise ValueError(f"{target} chronological split lacks both classes")
    model.fit(fit_frame[feature_columns], fit_frame[target])
    # scikit-learn 1.6+ represents a prefit estimator explicitly.  Keeping
    # the estimator frozen makes the calibration-only holdout unable to
    # refit the CatBoost model or accidentally consume ranking rows.
    smallest_class = int(calibration_frame[target].value_counts().min())
    calibration_cv = min(5, smallest_class)
    if calibration_cv < 2:
        raise ValueError(f"{target} calibration holdout needs at least two rows per class")
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(model), method="sigmoid", cv=calibration_cv
    )
    calibrated.fit(calibration_frame[feature_columns], calibration_frame[target])
    pred = calibrated.predict(validation_frame[feature_columns]).ravel()
    metrics = {
        "macro_f1": float(f1_score(validation_frame[target], pred, average="macro")),
        "recall": float(recall_score(validation_frame[target], pred, average="macro")),
    }
    bundle = {
        "model": calibrated,
        "target": target,
        "feature_columns": feature_columns,
        "class_order": [str(value) for value in model.classes_],
        "metrics": metrics,
        "training_rows": int(len(fit_frame)),
        "calibration_rows": int(len(calibration_frame)),
        "validation_rows": int(len(validation_frame)),
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
