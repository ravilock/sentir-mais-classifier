from __future__ import annotations

from app.classifier import FeelingClassifier
from app.settings import Settings


def test_classifier_loads_pipeline_once(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_pipeline(**kwargs):
        calls.append(kwargs)

        class FakePipeline:
            def __call__(self, **_kwargs):
                return {
                    "labels": ["anxious", "tense"],
                    "scores": [0.92, 0.71],
                }

        return FakePipeline()

    monkeypatch.setattr("app.classifier.pipeline", fake_pipeline)
    monkeypatch.setattr("app.classifier.torch.cuda.is_available", lambda: False)

    classifier = FeelingClassifier(Settings(model_name="fake-model"))

    classifier.load()
    classifier.load()

    assert classifier.is_loaded is True
    assert len(calls) == 1
    assert calls[0]["task"] == "zero-shot-classification"
    assert calls[0]["model"] == "fake-model"
    assert calls[0]["device"] == -1


def test_classifier_returns_ranked_scores(monkeypatch) -> None:
    class FakePipeline:
        def __call__(self, **kwargs):
            assert kwargs["sequences"] == "I feel anxious and tense."
            assert kwargs["candidate_labels"] == ["anxious", "tense", "sad"]
            assert kwargs["multi_label"] is True
            assert kwargs["hypothesis_template"] == "This text expresses {}."
            return {
                "labels": ["anxious", "tense", "sad"],
                "scores": [0.91, 0.84, 0.22],
            }

    monkeypatch.setattr("app.classifier.pipeline", lambda **kwargs: FakePipeline())
    monkeypatch.setattr("app.classifier.torch.cuda.is_available", lambda: False)

    classifier = FeelingClassifier(
        Settings(
            model_name="fake-model",
            default_labels=["anxious", "tense", "sad"],
        )
    )

    response = classifier.classify(
        text="I feel anxious and tense.",
        candidate_labels=None,
        top_k=2,
        multi_label=True,
        hypothesis_template=None,
    )

    assert response.model_name == "fake-model"
    assert response.primary_feeling.label == "anxious"
    assert response.primary_feeling.confidence == 0.91
    assert [score.label for score in response.secondary_feelings] == ["tense"]
    assert [score.label for score in response.all_scores] == ["anxious", "tense"]
