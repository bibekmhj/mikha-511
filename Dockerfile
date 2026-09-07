# Mikha-Ref v0.1 container.
#
# Builds a lean runtime image and serves the reference dashboard via
# Uvicorn on port 8000. Model dependencies (ultralytics + torch) are
# NOT installed here — the M5 dashboard runs end-to-end on the pure-
# Python persistence + decision + SQLite + FastAPI stack. Add the
# `[model]` extra downstream when running with a live detector.

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MIKHA_STORE=/data/events.sqlite

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --upgrade pip && pip install -e .

# Event store lives in a mounted volume so container restarts keep
# their history.
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000
CMD ["uvicorn", "mikha.ref.api:app", "--host", "0.0.0.0", "--port", "8000"]
