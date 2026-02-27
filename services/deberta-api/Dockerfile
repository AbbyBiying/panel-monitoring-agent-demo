# -----------------------------------------------------------------------------
#   Builder — installs deps and downloads model weights (has internet access)
#
#   MODEL_BASE defaults to plain python:3.12.8-slim (cold build, downloads
#   weights from HuggingFace). Pass the pre-built base image to skip the
#   714MB download:
#     docker build --build-arg MODEL_BASE=<base-image> ...
# -----------------------------------------------------------------------------
ARG MODEL_BASE=python:3.12.8-slim
FROM ${MODEL_BASE} AS builder

ENV APP_DIR=/app \
    HF_HOME=/app/.cache/huggingface

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /bin/uv

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR ${APP_DIR}

ADD pyproject.toml ${APP_DIR}/
ADD uv.lock ${APP_DIR}/

RUN set -eEx; \
    apt-get update && \
    apt-get upgrade -y openssl ca-certificates && \
    apt-get install -y --no-install-recommends procps ca-certificates && \
    update-ca-certificates && \
    uv sync --frozen --no-dev --no-install-project && \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
    rm -rf /var/lib/apt/lists/*

# Download model weights into builder cache
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='protectai/deberta-v3-base-prompt-injection-v2')"

# -----------------------------------------------------------------------------
#   Production — fully offline, copies venv and model cache from builder
# -----------------------------------------------------------------------------
FROM python:3.12.8-slim AS production

EXPOSE 8080

ENV APP_DIR=/app \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR ${APP_DIR}

RUN apt-get update && \
    apt-get install -y --no-install-recommends procps ca-certificates && \
    apt-get upgrade -y openssl ca-certificates && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos "" deberta-svc \
    && mkdir -p ${APP_DIR}/.cache/huggingface \
    && chown -R deberta-svc:deberta-svc ${APP_DIR}

# Copy virtualenv and model cache from builder
COPY --from=builder --chown=deberta-svc:deberta-svc /app/.venv /app/.venv
COPY --from=builder --chown=deberta-svc:deberta-svc /app/.cache/huggingface /app/.cache/huggingface

COPY --chown=deberta-svc:deberta-svc panel_monitoring ${APP_DIR}/panel_monitoring
COPY --chown=deberta-svc:deberta-svc services/deberta-api/main.py ${APP_DIR}/

USER deberta-svc

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "main:app"]
