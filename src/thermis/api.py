from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from thermis.artifacts import load_bundle
from thermis.features import FEATURE_COLUMNS
from thermis.fusion import FusionInput, RiskPolicy, fuse_prediction


class PredictionRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timestamp_utc: datetime
    frp: float = Field(ge=0)
    brightness_temperature: float | None = None
    frp_uncertainty: float | None = Field(default=None, ge=0)


class ModelRuntime:
    def __init__(
        self, stage1: dict[str, Any], stage2: dict[str, Any], demo: dict[str, Any] | None = None
    ):
        self.stage1 = stage1
        self.stage2 = stage2
        self.demo = demo or {}
        self.model_version = "tabular-local-0.1"

    @classmethod
    def from_paths(cls, root: Path, demo_path: Path | None = None) -> "ModelRuntime":
        demo = {}
        if demo_path and demo_path.exists():
            import json

            demo = json.loads(demo_path.read_text(encoding="utf-8"))
        return cls(
            load_bundle(root / "stage1.joblib"),
            load_bundle(root / "stage2.joblib"),
            demo=demo,
        )

    def predict(self, request: PredictionRequest) -> dict[str, Any]:
        values = {column: 0.0 for column in FEATURE_COLUMNS}
        values.update(
            latitude=request.latitude,
            longitude=request.longitude,
            frp=request.frp,
            brightness_temperature=request.brightness_temperature or 0.0,
            frp_uncertainty=request.frp_uncertainty or 0.0,
        )
        features = pd.DataFrame([values])[self.stage2["feature_columns"]]
        stage1_model = self.stage1["model"]
        stage2_model = self.stage2["model"]
        stage1_probabilities = dict(
            zip(stage1_model.classes_, stage1_model.predict_proba(features)[0], strict=True)
        )
        stage2_probabilities = dict(
            zip(stage2_model.classes_, stage2_model.predict_proba(features)[0], strict=True)
        )
        fused = fuse_prediction(
            FusionInput(
                actionable_probability=float(
                    stage1_probabilities.get("emergency_or_transient_fire", 0.0)
                ),
                tabular_stage_2={
                    str(key): float(value) for key, value in stage2_probabilities.items()
                },
                thermal_severity=min(1.0, request.frp / 100.0),
                context_completeness=0.8,
            ),
            RiskPolicy(),
        )
        return {
            "final_class": fused.final_class,
            "confidence": fused.confidence,
            "risk_score": fused.risk_score,
            "risk_level": fused.risk_level,
            "review_required": fused.review_required,
            "probabilities": fused.probabilities,
            "model_version": self.model_version,
        }

    def event(self, event_id: str) -> dict[str, Any] | None:
        return self.demo.get(event_id)


def create_app(runtime: ModelRuntime | Any | None = None) -> FastAPI:
    app = FastAPI(title="THERMIS SIH26162", version="0.1.0")
    active_runtime = runtime

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_version": getattr(active_runtime, "model_version", "demo"),
            "offline_demo": bool(getattr(active_runtime, "demo", {})),
        }

    @app.get("/events")
    def events() -> dict[str, Any]:
        demo = getattr(active_runtime, "demo", {})
        return {"events": list(demo.values())}

    @app.get("/events/{event_id}")
    def event(event_id: str) -> dict[str, Any]:
        value = active_runtime.event(event_id) if active_runtime else None
        if value is None:
            raise HTTPException(status_code=404, detail="event not found")
        return value

    @app.post("/predict")
    def predict(request: PredictionRequest) -> dict[str, Any]:
        if active_runtime is None:
            raise HTTPException(status_code=503, detail="model runtime is not loaded")
        return {"prediction": active_runtime.predict(request)}

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return getattr(active_runtime, "metrics", {"status": "demo metrics unavailable"})

    @app.get("/timeline/{event_id}")
    def timeline(event_id: str) -> dict[str, Any]:
        value = active_runtime.event(event_id) if active_runtime else None
        if value is None:
            raise HTTPException(status_code=404, detail="event not found")
        return {"event_id": event_id, "timeline_90d": value.get("timeline_90d", [])}

    return app
