from __future__ import annotations

import os
from dataclasses import dataclass, field


DEFAULT_LABELS = [
    "happy",
    "sad",
    "stressed",
    "angry",
    "doubtful",
    "anxious",
    "relaxed",
    "tense",
]


def _parse_csv(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default

    values = [value.strip() for value in raw.split(",")]
    return [value for value in values if value]


@dataclass(frozen=True)
class Settings:
    app_name: str = "sentir-mais-classifier"
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8010")))
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    model_name: str = field(
        default_factory=lambda: os.getenv(
            "MODEL_NAME",
            "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        )
    )
    default_labels: list[str] = field(
        default_factory=lambda: _parse_csv("DEFAULT_LABELS", DEFAULT_LABELS)
    )
    hypothesis_template: str = field(
        default_factory=lambda: os.getenv(
            "HYPOTHESIS_TEMPLATE",
            "This text expresses {}.",
        )
    )
    trust_remote_code: bool = field(
        default_factory=lambda: os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"
    )
    model_cache_dir: str | None = field(
        default_factory=lambda: os.getenv("MODEL_CACHE_DIR") or None
    )
