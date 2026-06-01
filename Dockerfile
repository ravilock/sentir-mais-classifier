FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ARG PIP_EXTRA_INDEX_URL=""

COPY pyproject.toml README.md ./
COPY app ./app

RUN pip install --no-cache-dir --upgrade pip \
    && if [ -n "$PIP_EXTRA_INDEX_URL" ]; then pip install --no-cache-dir --extra-index-url "$PIP_EXTRA_INDEX_URL" .; else pip install --no-cache-dir .; fi

EXPOSE 8010

CMD ["sentir-mais-classifier"]
