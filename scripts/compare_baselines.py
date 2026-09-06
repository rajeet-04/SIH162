"""Task 8 baseline comparison: logreg / RF / HistGradientBoosting vs CatBoost.

Reuses the exact development frame and chronological 80/20 split geometry of
thermis.training._fit_one (ranking rows excluded, supervised-eligible only).
No new dependencies (sklearn ships with the project). Writes
reports/baseline_comparison.json. A rules baseline is intentionally omitted:
labels are rule outputs, so it would score trivially by construction.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from thermis.artifacts import load_bundle
from thermis.features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "reports" / "baseline_comparison.json"
SEED = 26162

# Features the label rules read directly; removing them measures how much of
# the score is rule reconstruction vs independent signal.
RULE_FEATURES = frozenset(
    {"prior_detections_90d", "nearest_flare_distance_m", "nearest_industrial_distance_m"}
)


def dev_frame(target: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    features = pd.read_parquet(ROOT / "data/features/event_features.parquet")
    splits = pd.read_parquet(ROOT / "data/splits/event_splits.parquet")[["event_id", "split"]]
    labels = pd.read_parquet(ROOT / "data/labels/events_labeled.parquet")[
        ["event_id", "label_stage_1", "label_stage_2", "supervised_eligible"]
    ]
    frame = features.merge(splits, on="event_id").merge(labels, on="event_id")
    available = [c for c in FEATURE_COLUMNS if c in frame.columns]
    dev = frame[(frame["split"] != "ranking") & frame["supervised_eligible"]].dropna(
        subset=[target]
    )
    dev = dev.sort_values("timestamp_utc")
    cutoff = max(1, min(len(dev) - 1, int(len(dev) * 0.80)))
    fit, val = dev.iloc[:cutoff], dev.iloc[cutoff:]
    return fit[available], fit[target], val[available], val[target]


def drop_rule_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[c for c in frame.columns if c in RULE_FEATURES])


def score(name: str, model, x_fit, y_fit, x_val, y_val) -> dict:
    model.fit(x_fit, y_fit)
    pred = model.predict(x_val)
    return {"model": name, "macro_f1": float(f1_score(y_val, pred, average="macro"))}


def main() -> None:
    results: dict = {"seed": SEED, "note": "rules baseline omitted: labels are rule outputs"}
    for stage, target in (("stage1", "label_stage_1"), ("stage2", "label_stage_2")):
        x_fit, y_fit, x_val, y_val = dev_frame(target)
        scaler = StandardScaler().fit(x_fit)
        scaled_fit, scaled_val = scaler.transform(x_fit), scaler.transform(x_val)
        rows = [
            score("logreg", LogisticRegression(max_iter=2000),
                  scaled_fit, y_fit, scaled_val, y_val),
            score("random_forest",
                  RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
                  x_fit, y_fit, x_val, y_val),
            score("hist_gbm", HistGradientBoostingClassifier(random_state=SEED),
                  x_fit, y_fit, x_val, y_val),
            score("hist_gbm_no_rule_feats",
                  HistGradientBoostingClassifier(random_state=SEED),
                  drop_rule_features(x_fit), y_fit, drop_rule_features(x_val), y_val),
        ]
        bundle = load_bundle(ROOT / "models/tabular" / f"{stage}.joblib")
        rows.append({"model": "catboost_shipped", "macro_f1": bundle["metrics"]["macro_f1"]})
        rows.sort(key=lambda r: r["macro_f1"], reverse=True)
        results[stage] = {"rows": rows, "val_rows": len(y_val), "winner": rows[0]["model"]}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
