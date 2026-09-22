# syntax=docker/dockerfile:1
# Multi-stage: builder resolves/installs everything with uv (frozen lock),
# runtime ships only the venv + source. Both stages share the same CUDA base
# so the copied virtualenv keeps working (same interpreter paths).
#
# NOTE: gcc stays in the runtime image on purpose — torch>=2.14 dispatches
# some ops to Triton kernels, and Triton JIT-compiles a small C shim at
# inference time (its own ptxas is bundled; only the C compiler is missing
# from the slim CUDA image). Without it every stream fails at the first
# generate() with "Failed to find C compiler".
ARG CUDA_IMAGE=nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04

FROM ${CUDA_IMAGE} AS builder
ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PYTHON=/usr/bin/python3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/
WORKDIR /app
# Dependencies first (cached layer), source afterwards.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --extra asr --extra diarization
COPY src ./src
RUN uv sync --frozen --no-dev --extra asr --extra diarization

FROM ${CUDA_IMAGE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    UV_PYTHON=/usr/bin/python3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    libsndfile1 \
    ffmpeg \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /cache/huggingface \
    && chown -R app:app /cache/huggingface
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src
COPY --chown=app:app pyproject.toml ./
USER app
EXPOSE 8000
# Model loading takes a while on first boot: generous start-period.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
ENTRYPOINT ["/app/.venv/bin/transcript-server"]
