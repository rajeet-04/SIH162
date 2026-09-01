import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class LabelDecision:
    stage_1: str
    stage_2: str
    source: str
    confidence: float
    supervised_eligible: bool
    provenance: dict[str, Any]


def _rules(path: Path | None = None) -> dict[str, float]:
    rule_path = path or Path("config/label_rules.yaml")
    if not rule_path.exists():
        return {
            "flare_distance_m": 1000.0,
            "industrial_distance_m": 1500.0,
            "persistent_count_90d": 10.0,
            "stationary_spread_rate_km_h": 0.1,
            "minimum_supervised_confidence": 0.70,
        }
    return {key: float(value) for key, value in yaml.safe_load(rule_path.read_text()).items()}


def assign_event_label(row: pd.Series, rule_path: Path | None = None) -> LabelDecision:
    rules = _rules(rule_path)
    provenance = {
        key: row[key]
        for key in (
            "fire_ID",
            "fire_id",
            "fire_atlas_year",
            "landcover",
            "start_date",
            "end_date",
            "duration",
            "expansion",
            "fire_line",
            "speed",
            "direction",
            "direction_s",
            "label_source",
        )
        if key in row and pd.notna(row[key])
    }
    source_label = str(row.get("source_label", "")).lower()
    prior_count = float(row.get("prior_detections_90d", 0) or 0)
    flare_distance = float(row.get("nearest_flare_distance_m", float("inf")) or float("inf"))
    industrial_distance = float(
        row.get("nearest_industrial_distance_m", float("inf")) or float("inf")
    )
    spread_rate = float(row.get("spread_rate_km_h", 0) or 0)
    burned = bool(row.get("modis_burned_overlap", False))

    if source_label in {"manual_industrial", "authoritative_industrial"}:
        decision = ("emergency_or_transient_fire", "industrial_fire", "authoritative", 1.0)
    elif (
        prior_count >= rules["persistent_count_90d"]
        and flare_distance <= rules["flare_distance_m"]
        and spread_rate <= rules["stationary_spread_rate_km_h"]
    ):
        decision = (
            "persistent_or_non_emergency_heat",
            "persistent_industrial_heat_or_flare",
            "known_flare_and_persistence",
            0.90,
        )
    elif source_label in {"fire", "wildfire", "agricultural_burn"} or burned:
        decision = (
            "emergency_or_transient_fire",
            "wildfire_or_agricultural_burn",
            "fire_atlas_or_burned_area",
            0.85,
        )
    elif industrial_distance <= rules["industrial_distance_m"]:
        decision = (
            "emergency_or_transient_fire",
            "industrial_fire_candidate",
            "industrial_proximity",
            0.72,
        )
    else:
        decision = ("uncertain", "uncertain_review_required", "insufficient_evidence", 0.40)
    eligible = decision[3] >= rules["minimum_supervised_confidence"]
    return LabelDecision(*decision, supervised_eligible=eligible, provenance=provenance)


def assign_labels(events: pd.DataFrame, context: object | None = None) -> pd.DataFrame:
    """Apply conservative hierarchical labels and retain evidence provenance."""
    del context
    decisions = [assign_event_label(row) for _, row in events.iterrows()]
    result = events.copy()
    result["label_stage_1"] = [decision.stage_1 for decision in decisions]
    result["label_stage_2"] = [decision.stage_2 for decision in decisions]
    result["label_source"] = [decision.source for decision in decisions]
    result["label_confidence"] = [decision.confidence for decision in decisions]
    result["supervised_eligible"] = [decision.supervised_eligible for decision in decisions]
    result["label_provenance"] = [
        json.dumps(decision.provenance, sort_keys=True, default=str) for decision in decisions
    ]
    return result
