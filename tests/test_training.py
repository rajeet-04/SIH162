import numpy as np
import pandas as pd

from thermis.training import persistent_false_alert_rate, train_tabular, weighted_ranking_score


def test_persistent_false_alert_rate() -> None:
    truth = np.array(["persistent", "persistent", "active", "active"])
    pred = np.array(["active", "persistent", "active", "active"])
    assert persistent_false_alert_rate(truth, pred) == 0.5


def test_weighted_ranking_score_uses_approved_weights() -> None:
    metrics = {
        "industrial_recall": 0.9,
        "macro_f1": 0.8,
        "persistent_specificity": 0.85,
        "pr_auc": 0.88,
        "calibration_quality": 0.9,
        "speed_quality": 1.0,
    }
    assert round(weighted_ranking_score(metrics), 4) == 0.8795


def test_tabular_training_keeps_ranking_out_of_fit(tmp_path) -> None:
    rows = 30
    frame = pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(rows)],
            "timestamp_utc": pd.date_range("2024-01-01", periods=rows, tz="UTC"),
            "split": ["development"] * 27 + ["ranking"] * 3,
            "supervised_eligible": [True] * rows,
            "label_stage_1": ["active", "persistent"] * 15,
            "label_stage_2": ["industrial", "persistent"] * 15,
            "latitude": np.linspace(10, 12, rows),
            "longitude": np.linspace(70, 72, rows),
            "frp": np.linspace(1, 30, rows),
            "brightness_temperature": np.linspace(300, 330, rows),
            "frp_uncertainty": np.ones(rows),
            "prior_detections_7d": np.arange(rows),
            "prior_detections_30d": np.arange(rows),
            "prior_detections_90d": np.arange(rows),
            "stationary_count_90d": np.arange(rows),
            "nearest_flare_distance_m": np.linspace(1, 100, rows),
            "nearest_industrial_distance_m": np.linspace(1, 100, rows),
        }
    )
    bundles = train_tabular(frame, tmp_path)
    assert bundles["stage1"]["ranking_rows_used"] == 0
    assert bundles["stage1"]["validation_rows"] > 0
