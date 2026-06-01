from __future__ import annotations

from fastapi.testclient import TestClient

from app import main
from app.schemas import ClassifyResponse, FeelingScore
from app.settings import Settings


class FakeClassifier:
    def __init__(self) -> None:
        self.load_calls = 0
        self.classify_calls = []
        self.is_loaded = False
        self.model_name = "fake-model"
        self.default_labels = ["happy", "sad"]

    def load(self) -> None:
        self.load_calls += 1
        self.is_loaded = True

    def classify(self, **kwargs) -> ClassifyResponse:
        self.classify_calls.append(kwargs)
        return ClassifyResponse(
            text=kwargs["text"],
            primary_feeling=FeelingScore(label="happy", confidence=0.75),
            secondary_feelings=[FeelingScore(label="sad", confidence=0.2)],
            all_scores=[
                FeelingScore(label="happy", confidence=0.75),
                FeelingScore(label="sad", confidence=0.2),
            ],
            model_name=self.model_name,
            labels_used=kwargs["candidate_labels"] or self.default_labels,
            multi_label=kwargs["multi_label"],
        )


def test_healthcheck_reports_loaded_model(monkeypatch) -> None:
    fake_classifier = FakeClassifier()
    monkeypatch.setattr(main, "settings", Settings())
    monkeypatch.setattr(main, "classifier", fake_classifier)

    with TestClient(main.app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_name": "fake-model",
        "default_labels": ["happy", "sad"],
    }
    assert fake_classifier.load_calls == 1


def test_classify_requires_api_key_when_configured(monkeypatch) -> None:
    fake_classifier = FakeClassifier()
    monkeypatch.setattr(main, "settings", Settings(api_key="secret-key"))
    monkeypatch.setattr(main, "classifier", fake_classifier)

    with TestClient(main.app) as client:
        response = client.post("/classify", json={"text": "hello"})

    assert response.status_code == 401
    assert response.json() == {"detail": "unauthorized"}


def test_classify_accepts_request_with_api_key(monkeypatch) -> None:
    fake_classifier = FakeClassifier()
    monkeypatch.setattr(main, "settings", Settings(api_key="secret-key"))
    monkeypatch.setattr(main, "classifier", fake_classifier)

    with TestClient(main.app) as client:
        response = client.post(
            "/classify",
            headers={"Authorization": "secret-key"},
            json={
                "text": "I feel happy",
                "candidate_labels": ["happy", "sad"],
                "top_k": 2,
                "multi_label": True,
            },
        )

    assert response.status_code == 200
    assert response.json()["primary_feeling"] == {
        "label": "happy",
        "confidence": 0.75,
    }
    assert fake_classifier.classify_calls == [
        {
            "text": "I feel happy",
            "candidate_labels": ["happy", "sad"],
            "top_k": 2,
            "multi_label": True,
            "hypothesis_template": None,
        }
    ]
