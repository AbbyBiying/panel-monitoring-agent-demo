# panel_monitoring/app/retry.py
"""Shared retry configs using tenacity for external API calls."""

from __future__ import annotations

import logging

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Firestore: fast retries for transient network blips
firestore_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)

# LLM API: slower retries for rate limits and 503s
llm_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)

# Embedding API: similar to LLM (Google API call)
embedding_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)


async def firestore_write_with_retry(doc_ref, data, merge=False):
    """Firestore document write with exponential backoff."""

    @firestore_retry
    async def _write():
        await doc_ref.set(data, merge=merge)

    await _write()
