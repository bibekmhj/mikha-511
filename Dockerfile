# Mikha-511 reference container.
# NOTE: intentionally does not install model dependencies at M0. Ultralytics
# and torch are pulled in by the `model` extra when Mikha-Ref (M5) needs them.

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps kept minimal at M0. OpenCV headless needs libgl in some
# distros, but python:3.12-slim + opencv-python-headless is fine without it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --upgrade pip \
    && pip install -e .

# Dashboard entrypoint is defined in M5. At M0 the container just verifies
# that the package imports.
CMD ["python", "-c", "import mikha; print(f'mikha {mikha.__version__} ok')"]
