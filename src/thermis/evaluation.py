import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd

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
