import numpy as np

from thermis.training import persistent_false_alert_rate, weighted_ranking_score


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
