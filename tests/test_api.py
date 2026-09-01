from datetime import datetime, timezone

from fastapi.testclient import TestClient

from thermis.api import PredictionRequest, create_app


class FakeRuntime:
    model_version = "fake-1"
    demo = {
        "demo-industrial-fire": {
            "event_id": "demo-industrial-fire",
            "prediction": {"final_class": "industrial_fire", "model_version": "fake-1"},
            "evidence": {"thermal_change": 0.9},
            "timeline_90d": [{"days_ago": 0, "frp": 80}],
        }
    }
    metrics = {"ranking_set": "offline-demo"}

    def event(self, event_id: str):
        return self.demo.get(event_id)

    def predict(self, request: PredictionRequest):
        return {"final_class": "industrial_fire", "model_version": self.model_version}


def test_demo_event_contains_evidence_and_timeline() -> None:
    client = TestClient(create_app(FakeRuntime()))
    response = client.get("/events/demo-industrial-fire")
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"]["final_class"] == "industrial_fire"
    assert payload["prediction"]["model_version"]
    assert payload["evidence"]
    assert payload["timeline_90d"]


def test_missing_critical_fields_returns_422() -> None:
    client = TestClient(create_app(FakeRuntime()))
    assert client.post("/predict", json={"latitude": 20.0}).status_code == 422


def test_predict_accepts_valid_request() -> None:
    client = TestClient(create_app(FakeRuntime()))
    response = client.post(
        "/predict",
        json={
            "latitude": 20.0,
            "longitude": 70.0,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "frp": 12.0,
        },
    )
    assert response.status_code == 200
    assert response.json()["prediction"]["model_version"] == "fake-1"
