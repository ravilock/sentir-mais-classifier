from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import ClassifyRequest


def test_classify_request_accepts_valid_payload() -> None:
    request = ClassifyRequest(
        text="I feel anxious after the meeting.",
        candidate_labels=["anxious", "tense"],
        top_k=2,
        multi_label=True,
    )

    assert request.text == "I feel anxious after the meeting."
    assert request.candidate_labels == ["anxious", "tense"]


def test_classify_request_rejects_blank_text() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ClassifyRequest(text="   ")

    assert "text must not be blank" in str(exc_info.value)


def test_classify_request_rejects_blank_candidate_labels() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ClassifyRequest(text="hello", candidate_labels=["   "])

    assert "candidate_labels must contain at least one non-blank label" in str(exc_info.value)
