from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from thermis.artifacts import load_bundle
from thermis.features import FEATURE_COLUMNS
from thermis.fusion import FusionInput, RiskPolicy, fuse_prediction
from thermis.image_model import _image_tensor, inspect_image


class PredictionRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timestamp_utc: datetime
    frp: float = Field(ge=0)
    brightness_temperature: float | None = None
    frp_uncertainty: float | None = Field(default=None, ge=0)


class ImageVerificationRequest(BaseModel):
    path: str = Field(min_length=1)


class ModelRuntime:
    def __init__(
        self,
        stage1: dict[str, Any],
        stage2: dict[str, Any],
        demo: dict[str, Any] | None = None,
        image_model_path: Path | None = None,
    ):
        self.stage1 = stage1
        self.stage2 = stage2
        self.demo = demo or {}
        self.model_version = "tabular-local-0.1"
        self.image_model = None
        self.image_metadata: dict[str, Any] = {}
        if image_model_path and image_model_path.exists():
            import json

            import torch

            self.image_model = torch.jit.load(str(image_model_path), map_location="cpu").eval()
            metadata_path = image_model_path.with_name("metadata.json")
            if metadata_path.exists():
                self.image_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.metrics = {
            "stage1": stage1.get("metrics", {}),
            "stage2": stage2.get("metrics", {}),
            "ranking_promotion": "blocked_pending_authoritative_labels",
        }

    @classmethod
    def from_paths(
        cls,
        root: Path,
        demo_path: Path | None = None,
        image_model_path: Path | None = None,
    ) -> "ModelRuntime":
        demo = {}
        if demo_path and demo_path.exists():
            import json

            demo = json.loads(demo_path.read_text(encoding="utf-8"))
        return cls(
            load_bundle(root / "stage1.joblib"),
            load_bundle(root / "stage2.joblib"),
            demo=demo,
            image_model_path=image_model_path or root.parent / "image-smoke/image_verifier.ts",
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

    def verify_image(self, path: Path) -> dict[str, Any]:
        if self.image_model is None:
            raise RuntimeError("image verifier is not loaded")
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)
        quality = inspect_image(path)
        if not quality.readable or quality.channels is None:
            raise ValueError(f"image is not readable: {quality.reason or 'unknown error'}")
        import torch

        with torch.inference_mode():
            logits = self.image_model(_image_tensor(path).unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)[0].tolist()
        classes = self.image_metadata.get("classes", ["fire", "no_fire"])
        return {
            "path": str(path),
            "classes": {
                str(label): round(float(probability), 6)
                for label, probability in zip(classes, probabilities, strict=True)
            },
            "predicted_class": str(classes[int(torch.argmax(logits, dim=1).item())]),
            "model_version": "image-resnet18-local-0.1",
        }


def create_app(runtime: ModelRuntime | Any | None = None) -> FastAPI:
    app = FastAPI(title="THERMIS SIH26162", version="0.1.0")
    active_runtime = runtime

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_version": getattr(active_runtime, "model_version", "demo"),
            "offline_demo": bool(getattr(active_runtime, "demo", {})),
            "image_verifier_loaded": bool(getattr(active_runtime, "image_model", None)),
            "image_verifier_device": "cpu",
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

    @app.post("/verify-image")
    def verify_image(request: ImageVerificationRequest) -> dict[str, Any]:
        if active_runtime is None or not hasattr(active_runtime, "verify_image"):
            raise HTTPException(status_code=503, detail="image verifier is not loaded")
        try:
            return {"verification": active_runtime.verify_image(Path(request.path))}
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=f"image not found: {error}") from error
        except (OSError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return getattr(active_runtime, "metrics", {"status": "demo metrics unavailable"})

    @app.get("/timeline/{event_id}")
    def timeline(event_id: str) -> dict[str, Any]:
        value = active_runtime.event(event_id) if active_runtime else None
        if value is None:
            raise HTTPException(status_code=404, detail="event not found")
        return {"event_id": event_id, "timeline_90d": value.get("timeline_90d", [])}

    @app.get("/replay")
    def replay(count: int = 50, cursor: int = 0, seed: int = 26162) -> dict[str, Any]:
        """Stream real labeled events with live model verdicts (replay feed).

        Rows come from the prepared event table in deterministic shuffled order;
        each row is scored by the loaded tabular models at request time.
        """
        if active_runtime is None:
            raise HTTPException(status_code=503, detail="model runtime is not loaded")
        count = max(1, min(200, count))
        table = pd.read_parquet(
            Path("data/labels/events_labeled.parquet"),
            columns=[
                "event_id", "latitude", "longitude", "timestamp_utc", "frp",
                "brightness_temperature", "frp_uncertainty", "prior_detections_7d",
                "prior_detections_30d", "prior_detections_90d",
                "nearest_flare_distance_m", "label_source",
            ],
        ).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        window = table.iloc[cursor : cursor + count]
        items = [replay_item(active_runtime, row) for row in window.itertuples()]
        return {"events": items, "total": len(table), "cursor": cursor, "count": len(items)}

    return app


def replay_item(runtime: ModelRuntime | Any, row: Any) -> dict[str, Any]:
    """Score one stored event row through the live tabular models."""
    request = PredictionRequest(
        latitude=float(row.latitude),
        longitude=float(row.longitude),
        timestamp_utc=pd.Timestamp(row.timestamp_utc).to_pydatetime(),
        frp=float(row.frp or 0.0),
        brightness_temperature=float(row.brightness_temperature or 0.0),
        frp_uncertainty=float(row.frp_uncertainty or 0.0),
    )
    prediction = runtime.predict(request)
    return {
        "event_id": f"replay-{row.event_id}",
        "latitude": float(row.latitude),
        "longitude": float(row.longitude),
        "prediction": prediction,
        "evidence": {
            "frp": float(row.frp or 0.0),
            "prior_detections_7d": int(row.prior_detections_7d or 0),
            "prior_detections_30d": int(row.prior_detections_30d or 0),
            "prior_detections_90d": int(row.prior_detections_90d or 0),
            "nearest_flare_distance_m": float(row.nearest_flare_distance_m or 0.0),
            "label_source": str(row.label_source),
        },
        "timeline_90d": [
            {"days_ago": 90, "prior_detections": 0},
            {"days_ago": 30, "prior_detections": int(row.prior_detections_30d or 0)},
            {"days_ago": 7, "prior_detections": int(row.prior_detections_7d or 0)},
            {"days_ago": 0, "frp": float(row.frp or 0.0)},
        ],
        "replay": True,
    }
