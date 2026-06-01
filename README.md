# sentir-mais-classifier

Python API service for feeling classification in Sentir Mais.

## What it does

- loads the classifier model into memory on startup
- exposes `POST /classify`
- exposes `GET /healthz`

Default model:

- `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`

Default labels:

- `happy`
- `sad`
- `stressed`
- `angry`
- `doubtful`
- `anxious`
- `relaxed`
- `tense`

## Run locally

Install:

```bash
pip install .
```

Run:

```bash
sentir-mais-classifier
```

Or:

```bash
python -m app.main
```

Default address:

- `http://localhost:8010`

## GPU inference

The service can force device selection with `MODEL_DEVICE`:

- `auto`: use CUDA when available, otherwise CPU
- `cpu`: always use CPU
- `cuda`: require the first CUDA device
- `cuda:0`, `cuda:1`, ...: require a specific CUDA device

If `MODEL_DEVICE` is set to `cuda` or `cuda:N` and no CUDA device is available, startup fails instead of silently falling back to CPU.

For a Docker image that installs the CUDA-enabled PyTorch wheel, build with:

```bash
docker build \
  --build-arg PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  -t sentir-mais-classifier:gpu .
```

Run it with GPU access:

```bash
docker run --rm --gpus all \
  -e MODEL_DEVICE=cuda \
  -p 8010:8010 \
  sentir-mais-classifier:gpu
```

## Published images

The GitHub workflow publishes two container variants to GHCR:

- `ghcr.io/ravilock/sentir-mais-classifier:latest`: default CPU-oriented image
- `ghcr.io/ravilock/sentir-mais-classifier:latest-gpu`: image built with the CUDA PyTorch wheel

Branch, tag, and SHA tags follow the same pattern, with the GPU variant using the `-gpu` suffix.

## Tests

Create a virtual environment if you want isolation:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dev dependencies:

```bash
pip install ".[dev]"
```

Run the full test suite:

```bash
pytest
```

Run a single test file:

```bash
pytest tests/test_main.py
```

## Environment

- `HOST`
- `PORT`
- `API_KEY`
- `MODEL_NAME`
- `DEFAULT_LABELS`
- `HYPOTHESIS_TEMPLATE`
- `MODEL_CACHE_DIR`
- `TRUST_REMOTE_CODE`
- `MODEL_DEVICE`

## API

### `GET /healthz`

Returns service and model readiness.

### `POST /classify`

If `API_KEY` is configured, requests must send:

```text
Authorization: <API_KEY>
```

Example request:

```json
{
  "text": "I had a fight at work and I feel tense and anxious about tomorrow.",
  "top_k": 3,
  "multi_label": true
}
```

Example response:

```json
{
  "text": "I had a fight at work and I feel tense and anxious about tomorrow.",
  "primary_feeling": {
    "label": "anxious",
    "confidence": 0.91
  },
  "secondary_feelings": [
    {
      "label": "tense",
      "confidence": 0.88
    },
    {
      "label": "stressed",
      "confidence": 0.74
    }
  ],
  "all_scores": [
    {
      "label": "anxious",
      "confidence": 0.91
    },
    {
      "label": "tense",
      "confidence": 0.88
    },
    {
      "label": "stressed",
      "confidence": 0.74
    }
  ],
  "model_name": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
  "labels_used": [
    "happy",
    "sad",
    "stressed",
    "angry",
    "doubtful",
    "anxious",
    "relaxed",
    "tense"
  ],
  "multi_label": true
}
```
