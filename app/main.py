from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, status

from app.classifier import FeelingClassifier
from app.schemas import ClassifyRequest, ClassifyResponse, HealthResponse
from app.settings import Settings

settings = Settings()
classifier = FeelingClassifier(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    classifier.load()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)


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
    return classifier.classify(
        text=request.text,
        candidate_labels=request.candidate_labels,
        top_k=request.top_k,
        multi_label=request.multi_label,
        hypothesis_template=request.hypothesis_template,
    )


def run() -> None:
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    run()
