# -----------------------------------------------------------------------------
#   Builder
# -----------------------------------------------------------------------------
FROM python:3.12.8-slim AS builder

ENV APP_DIR=/app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /bin/uv

WORKDIR ${APP_DIR}

ADD pyproject.toml ${APP_DIR}/
ADD uv.lock ${APP_DIR}/

RUN set -eEx; \
    apt-get update && \
    apt-get upgrade -y openssl ca-certificates && \
    apt-get install -y --no-install-recommends procps ca-certificates && \
    update-ca-certificates && \
    pip install -U pip && \
    uv sync --frozen --no-dev --system && \
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

ENV APP_DIR=/app
WORKDIR ${APP_DIR}

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /bin/uv

ADD pyproject.toml ${APP_DIR}/
ADD uv.lock ${APP_DIR}/

RUN adduser --disabled-password --gecos "" deberta-svc \
    && chown -R deberta-svc:deberta-svc ${APP_DIR} \
    && chmod -R 770 ${APP_DIR}

RUN apt-get update && \
    apt-get install -y --no-install-recommends procps ca-certificates && \
    apt-get upgrade -y openssl ca-certificates && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/* /usr/local/bin/
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Force test stage to complete before shipping
COPY --from=test /tests-passed /tests-passed

COPY panel_monitoring ${APP_DIR}/panel_monitoring
COPY services/deberta-api/main.py ${APP_DIR}/

RUN chmod -R 755 ${APP_DIR}

USER deberta-svc

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "main:app"]
