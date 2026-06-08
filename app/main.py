from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import time
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status

from app.classifier import FeelingClassifier
from app.schemas import ClassifyRequest, ClassifyResponse, HealthResponse
from app.settings import Settings

settings = Settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)
classifier = FeelingClassifier(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        "classifier service starting model=%s device=%s cache_dir=%s log_level=%s",
        settings.model_name,
        settings.model_device,
        settings.model_cache_dir,
        settings.log_level,
    )
    classifier.load()
    try:
        yield
    finally:
        logger.info("classifier service stopping")


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or request.headers.get("X-Amzn-Trace-Id") or uuid4().hex
    started_at = time.monotonic()
    client = request.client.host if request.client else "unknown"
    logger.info(
        "http request started request_id=%s method=%s path=%s client=%s",
        request_id,
        request.method,
        request.url.path,
        client,
    )

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "http request failed request_id=%s method=%s path=%s duration_ms=%s",
            request_id,
            request.method,
            request.url.path,
            int((time.monotonic() - started_at) * 1000),
        )
        raise

    response.headers["X-Request-ID"] = request_id
    logger.info(
        "http request completed request_id=%s method=%s path=%s status_code=%s duration_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        int((time.monotonic() - started_at) * 1000),
    )
    return response


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not settings.api_key:
        return

    if authorization != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthorized",
        )


@app.get("/healthz", response_model=HealthResponse)
async def healthcheck() -> HealthResponse:
    logger.debug(
        "healthcheck requested model_loaded=%s model_name=%s default_label_count=%s",
        classifier.is_loaded,
        classifier.model_name,
        len(classifier.default_labels),
    )
    return HealthResponse(
        status="ok",
        model_loaded=classifier.is_loaded,
        model_name=classifier.model_name,
        default_labels=classifier.default_labels,
    )


@app.post("/classify", response_model=ClassifyResponse)
async def classify(
    request: ClassifyRequest,
    _: None = Depends(require_api_key),
) -> ClassifyResponse:
    logger.info(
        "classify request received text_length=%s candidate_label_count=%s top_k=%s multi_label=%s",
        len(request.text),
        len(request.candidate_labels or classifier.default_labels),
        request.top_k,
        request.multi_label,
    )
    return classifier.classify(
        text=request.text,
        candidate_labels=request.candidate_labels,
        top_k=request.top_k,
        multi_label=request.multi_label,
        hypothesis_template=request.hypothesis_template,
    )


def run() -> None:
    logger.info(
        "starting classifier service host=%s port=%s model=%s log_level=%s",
        settings.host,
        settings.port,
        settings.model_name,
        settings.log_level,
    )
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    run()
