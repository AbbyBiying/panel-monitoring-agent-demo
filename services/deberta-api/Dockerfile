# -----------------------------------------------------------------------------
#   Builder
# -----------------------------------------------------------------------------
FROM python:3.12.8-slim AS builder

ENV APP_DIR=/app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /bin/uv

WORKDIR ${APP_DIR}

ADD pyproject.toml ${APP_DIR}/
ADD uv.lock ${APP_DIR}/

RUN set -eEx; \
    apt-get update && \
    apt-get upgrade -y openssl ca-certificates && \
    apt-get install -y --no-install-recommends procps ca-certificates && \
    update-ca-certificates && \
    uv sync --frozen --no-dev && \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
#   Test
# -----------------------------------------------------------------------------
FROM builder AS test

COPY panel_monitoring ${APP_DIR}/panel_monitoring
COPY services/deberta-api/main.py ${APP_DIR}/
COPY conftest.py ${APP_DIR}/
COPY tests/test_deberta_api.py ${APP_DIR}/tests/

# Build fails here if any test fails
RUN uv run pytest tests/test_deberta_api.py -v

RUN touch /tests-passed

# -----------------------------------------------------------------------------
#   Production
# -----------------------------------------------------------------------------
FROM python:3.12.8-slim AS production

EXPOSE 8080

ENV APP_DIR=/app \
    HF_HOME=/app/.cache/huggingface

WORKDIR ${APP_DIR}

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

RUN adduser --disabled-password --gecos "" deberta-svc \
    && mkdir -p ${APP_DIR}/.cache/huggingface \
    && chown -R deberta-svc:deberta-svc ${APP_DIR}

RUN apt-get update && \
    apt-get install -y --no-install-recommends procps ca-certificates && \
    apt-get upgrade -y openssl ca-certificates && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder instead of system site-packages
COPY --from=builder /app/.venv /app/.venv

# Force test stage to complete before shipping
COPY --from=test /tests-passed /tests-passed

COPY --chown=deberta-svc:deberta-svc panel_monitoring ${APP_DIR}/panel_monitoring
COPY --chown=deberta-svc:deberta-svc services/deberta-api/main.py ${APP_DIR}/

# Bake model weights into image to eliminate cold-start download
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='protectai/deberta-v3-base-prompt-injection-v2')" \
    && chown -R deberta-svc:deberta-svc ${APP_DIR}/.cache

USER deberta-svc

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "main:app"]
