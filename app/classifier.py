from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

import torch
from transformers import pipeline

from app.schemas import ClassifyResponse, FeelingScore
from app.settings import Settings


@dataclass
class ModelState:
    pipeline: Any | None = None


class FeelingClassifier:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._state = ModelState()
        self._lock = Lock()

    @property
    def model_name(self) -> str:
        return self._settings.model_name

    @property
    def default_labels(self) -> list[str]:
        return self._settings.default_labels

    @property
    def is_loaded(self) -> bool:
        return self._state.pipeline is not None

    def load(self) -> None:
        if self._state.pipeline is not None:
            return

        with self._lock:
            if self._state.pipeline is not None:
                return

            device = 0 if torch.cuda.is_available() else -1
            self._state.pipeline = pipeline(
                task="zero-shot-classification",
                model=self._settings.model_name,
                device=device,
                trust_remote_code=self._settings.trust_remote_code,
                model_kwargs={
                    "cache_dir": self._settings.model_cache_dir,
                },
            )

    def classify(
        self,
        *,
        text: str,
        candidate_labels: list[str] | None,
        top_k: int,
        multi_label: bool,
        hypothesis_template: str | None,
    ) -> ClassifyResponse:
        self.load()
        labels = candidate_labels or self._settings.default_labels
        template = hypothesis_template or self._settings.hypothesis_template

        result = self._state.pipeline(
            sequences=text,
            candidate_labels=labels,
            multi_label=multi_label,
            hypothesis_template=template,
        )

        ranked_scores = [
            FeelingScore(label=label, confidence=score)
            for label, score in zip(result["labels"], result["scores"], strict=True)
        ]

        limited_scores = ranked_scores[:top_k]
        primary = limited_scores[0]
        secondary = limited_scores[1:]

        return ClassifyResponse(
            text=text,
            primary_feeling=primary,
            secondary_feelings=secondary,
            all_scores=limited_scores,
            model_name=self._settings.model_name,
            labels_used=labels,
            multi_label=multi_label,
        )
