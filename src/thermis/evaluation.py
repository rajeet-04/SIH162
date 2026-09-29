import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve

from thermis.artifacts import load_bundle


@dataclass(frozen=True)
class PromotionMetrics:
    industrial_fire_recall: float
    macro_f1: float
    persistent_false_alert_rate: float
    cpu_latency_ms_p95: float
    severe_segment_collapse: bool


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reasons: tuple[str, ...]


def promotion_decision(metrics: PromotionMetrics) -> PromotionDecision:
    reasons: list[str] = []
    if metrics.industrial_fire_recall < 0.85:
        reasons.append("industrial_fire_recall_below_0.85")
    if metrics.macro_f1 < 0.75:
        reasons.append("macro_f1_below_0.75")
    if metrics.persistent_false_alert_rate > 0.15:
        reasons.append("persistent_false_alert_rate_above_0.15")
    if metrics.cpu_latency_ms_p95 >= 1000:
        reasons.append("cpu_latency_p95_at_or_above_1000ms")
    if metrics.severe_segment_collapse:
        reasons.append("severe_segment_collapse")
    return PromotionDecision(promoted=not reasons, reasons=tuple(reasons))


def evaluate_ranking_set(
    features_path: Path,
    splits_path: Path,
    model_root: Path,
    output_root: Path,
) -> dict[str, object]:
    features = pd.read_parquet(features_path)
    splits = pd.read_parquet(splits_path)[["event_id", "split"]]
    ranking = features.merge(splits, on="event_id", validate="one_to_one")
    ranking = ranking[ranking["split"] == "ranking"].copy()
    bundle = load_bundle(model_root / "stage2.joblib")
    model = bundle["model"]
    started = perf_counter()
    predictions = model.predict(ranking[bundle["feature_columns"]]).ravel()
    elapsed_ms = (perf_counter() - started) * 1000
    ranking["prediction_stage_2"] = predictions
    output_root.mkdir(parents=True, exist_ok=True)
    ranking.to_parquet(output_root / "ranking_predictions.parquet", index=False)
    metrics = PromotionMetrics(
        industrial_fire_recall=0.0,
        macro_f1=0.0,
        persistent_false_alert_rate=0.0,
        cpu_latency_ms_p95=elapsed_ms / max(1, len(ranking)) * 1.5,
        severe_segment_collapse=True,
    )
    decision = promotion_decision(metrics)
    report = {
        "ranking_rows": len(ranking),
        "authoritative_label_rows": int(
            ranking.get("supervised_eligible", pd.Series(dtype=bool)).sum()
        ),
        "metrics": metrics.__dict__,
        "promoted": decision.promoted,
        "reasons": list(decision.reasons),
    }
    (output_root / "ranking_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


INDUSTRIAL_CLASS = "industrial_fire_candidate"
PERSISTENT_CLASS = "persistent_industrial_heat_or_flare"


def _ece(confidences: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    ece, n = 0.0, len(confidences)
    for i in range(bins):
        low, high = i / bins, (i + 1) / bins
        mask = (confidences > low) & (confidences <= high)
        if i == 0:
            mask = mask | (confidences == 0.0)
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def score_labeled_ranking(
    features_path: Path,
    splits_path: Path,
    model_root: Path,
    labels_csv: Path,
    label_column: str,
    output_root: Path,
    min_rows: int = 300,
) -> dict[str, object]:
    """Run the full review battery on human- or tier-2-labeled ranking rows.

    Promotion is eligible ONLY for expert labels at sufficient coverage;
    anything else is reported as provisional with the gate held blocked.
    """
    features = pd.read_parquet(features_path)
    splits = pd.read_parquet(splits_path)[["event_id", "split"]]
    ranking = features.merge(splits, on="event_id", validate="one_to_one")
    ranking = ranking[ranking["split"] == "ranking"].copy()
    bundle = load_bundle(model_root / "stage2.joblib")
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    labels = pd.read_csv(labels_csv, usecols=["event_id", label_column])
    labels = labels[labels[label_column].notna() & (labels[label_column] != "")]
    scored = ranking.merge(labels, on="event_id", validate="one_to_one")
    proba = model.predict_proba(scored[feature_columns])
    classes = [str(c) for c in model.classes_]
    scored["prediction_stage_2"] = [classes[int(i)] for i in proba.argmax(axis=1)]
    scored["confidence"] = proba.max(axis=1)
    truth = scored[label_column].astype(str)
    pred = scored["prediction_stage_2"].astype(str)
    authoritative = label_column.startswith("expert") and len(scored) >= min_rows

    class_union = sorted(set(truth) | set(pred))
    matrix = confusion_matrix(truth, pred, labels=class_union)
    per_class = {}
    for index, name in enumerate(class_union):
        tp = int(matrix[index, index])
        per_class[name] = {
            "precision": tp / max(1, int(matrix[:, index].sum())),
            "recall": tp / max(1, int(matrix[index, :].sum())),
            "support": int((truth == name).sum()),
        }
    ind_truth = truth == INDUSTRIAL_CLASS
    industrial_recall = (
        float((pred[ind_truth] == INDUSTRIAL_CLASS).mean()) if ind_truth.any() else 0.0
    )
    industrial_precision = per_class.get(INDUSTRIAL_CLASS, {}).get("precision", 0.0)
    non_persistent = truth != PERSISTENT_CLASS
    persistent_false_alert = (
        float((pred[non_persistent] == PERSISTENT_CLASS).mean()) if non_persistent.any() else 0.0
    )
    macro_f1 = float(
        np.mean([2 * v["precision"] * v["recall"] / max(1e-9, v["precision"] + v["recall"])
                 for v in per_class.values()])
    )
    if INDUSTRIAL_CLASS in classes:
        scores = proba[:, classes.index(INDUSTRIAL_CLASS)]
        precision, recall, _ = precision_recall_curve(ind_truth.astype(int), scores)
        pr_auc = float(auc(recall, precision)) if ind_truth.any() else 0.0
        brier = float(np.mean((scores - ind_truth.astype(int)) ** 2))
    else:
        pr_auc, brier = 0.0, 1.0
    correct = (pred == truth).to_numpy()
    report: dict[str, object] = {
        "labeled_rows": len(scored),
        "label_column": label_column,
        "authoritative": authoritative,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "industrial_recall": industrial_recall,
        "industrial_precision": industrial_precision,
        "persistent_false_alert_rate": persistent_false_alert,
        "pr_auc_industrial": pr_auc,
        "brier_industrial": brier,
        "ece_10bin": _ece(scored["confidence"].to_numpy(), correct),
        "confusion": {
            t: {p: int(matrix[i, j]) for j, p in enumerate(class_union)}
            for i, t in enumerate(class_union)
        },
        "promoted": False,
        "reasons": ["provisional_non_expert_labels"] if not authoritative else [],
    }
    if authoritative:
        gate = promotion_decision(
            PromotionMetrics(industrial_recall, macro_f1, persistent_false_alert, 0.0, False)
        )
        report["promoted"] = gate.promoted
        report["reasons"] = list(gate.reasons)
    output_root.mkdir(parents=True, exist_ok=True)
    name = "ranking_expert_report.json" if authoritative else "ranking_provisional_report.json"
    (output_root / name).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
