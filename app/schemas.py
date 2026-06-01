from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1, description="Normalized context or extracted event text.")
    candidate_labels: list[str] | None = Field(
        default=None,
        description="Optional override for the target feeling labels.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum number of top labels to return.",
    )
    multi_label: bool = Field(
        default=True,
        description="Whether multiple labels may independently apply.",
    )
    hypothesis_template: str | None = Field(
        default=None,
        description="Optional zero-shot hypothesis template.",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text must not be blank")
        return normalized

    @field_validator("candidate_labels")
    @classmethod
    def validate_candidate_labels(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value

        normalized = [label.strip() for label in value if label.strip()]
        if not normalized:
            raise ValueError("candidate_labels must contain at least one non-blank label")
        return normalized


class FeelingScore(BaseModel):
    label: str
    confidence: float


class ClassifyResponse(BaseModel):
    text: str
    primary_feeling: FeelingScore
    secondary_feelings: list[FeelingScore]
    all_scores: list[FeelingScore]
    model_name: str
    labels_used: list[str]
    multi_label: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    default_labels: list[str]
