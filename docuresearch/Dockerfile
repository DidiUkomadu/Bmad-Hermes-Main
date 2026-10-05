# DocuResearch — production image (single process, SQLite on a mounted volume).
#
#   docker build -t docuresearch .
#   docker run -p 8000:8000 -v docuresearch-data:/data --env-file .env docuresearch
#
# See docs/deployment.md for HTTPS (docker compose + Caddy) and operations.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DOCURESEARCH_DB_PATH=/data/docuresearch.db

WORKDIR /app

# CPU-only PyTorch first: the default wheel bundles GPU libraries (several GB).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# Dependencies (from pyproject.toml), installed before the code for layer caching.
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > requirements.txt \
    && pip install -r requirements.txt

# Bake the embedding model into the image so the container never downloads it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY app ./app
COPY config.example.toml ./

RUN useradd --create-home --uid 10001 docuresearch \
    && mkdir -p /data && chown docuresearch:docuresearch /data
USER docuresearch

# Embedding model is local now; stay offline at runtime.
ENV HF_HUB_OFFLINE=1

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4)"

# One worker: SQLite plus an in-memory sign-in throttle assume a single process.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
