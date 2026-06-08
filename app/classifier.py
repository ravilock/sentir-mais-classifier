from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any

import torch
from transformers import pipeline

from app.schemas import ClassifyResponse, FeelingScore
from app.settings import Settings

logger = logging.getLogger(__name__)


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

    def _resolve_device(self) -> int | str:
        configured = self._settings.model_device.lower()

        if configured == "auto":
            return 0 if torch.cuda.is_available() else -1

        if configured == "cpu":
            return -1

        if configured == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("MODEL_DEVICE=cuda but no CUDA device is available")
            return 0

        if configured.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"MODEL_DEVICE={self._settings.model_device} but no CUDA device is available"
                )

            device_index = configured.removeprefix("cuda:")
            if not device_index.isdigit():
                raise ValueError(
                    f"Unsupported MODEL_DEVICE value: {self._settings.model_device}"
                )

            return int(device_index)

        raise ValueError(f"Unsupported MODEL_DEVICE value: {self._settings.model_device}")

    def load(self) -> None:
        if self._state.pipeline is not None:
            logger.debug("classifier model already loaded model=%s", self._settings.model_name)
            return

        with self._lock:
            if self._state.pipeline is not None:
                logger.debug("classifier model became loaded while waiting lock model=%s", self._settings.model_name)
                return

            resolved_device = self._resolve_device()
            logger.info(
                "classifier model load started model=%s device=%s cache_dir=%s trust_remote_code=%s",
                self._settings.model_name,
                resolved_device,
                self._settings.model_cache_dir,
                self._settings.trust_remote_code,
            )
            started_at = time.monotonic()
            self._state.pipeline = pipeline(
                task="zero-shot-classification",
                model=self._settings.model_name,
                device=resolved_device,
                trust_remote_code=self._settings.trust_remote_code,
                model_kwargs={
                    "cache_dir": self._settings.model_cache_dir,
                },
            )
            logger.info(
                "classifier model load completed model=%s device=%s duration_ms=%s",
                self._settings.model_name,
                resolved_device,
                int((time.monotonic() - started_at) * 1000),
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
        request_summary = {
            "text_length": len(text),
            "candidate_label_count": len(labels),
            "top_k": top_k,
            "multi_label": multi_label,
            "hypothesis_template": template,
        }
        logger.info("classifier inference started summary=%s", request_summary)
        started_at = time.monotonic()

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
        logger.info(
            "classifier inference completed summary=%s primary_feeling=%s duration_ms=%s",
            request_summary,
            primary.label,
            int((time.monotonic() - started_at) * 1000),
        )

        return ClassifyResponse(
            text=text,
            primary_feeling=primary,
            secondary_feelings=secondary,
            all_scores=limited_scores,
            model_name=self._settings.model_name,
            labels_used=labels,
            multi_label=multi_label,
        )
