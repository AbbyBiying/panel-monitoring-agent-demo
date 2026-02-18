# tests/test_retry.py
"""Unit tests for retry behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from panel_monitoring.app.retry import (
    embedding_retry,
    firestore_retry,
    firestore_write_with_retry,
    llm_retry,
)


# --- Decorator smoke tests ---


def test_firestore_retry_succeeds_on_first_try():
    @firestore_retry
    def fn():
        return "ok"

    assert fn() == "ok"


def test_llm_retry_succeeds_on_first_try():
    @llm_retry
    def fn():
        return "ok"

    assert fn() == "ok"


def test_embedding_retry_succeeds_on_first_try():
    @embedding_retry
    def fn():
        return "ok"

    assert fn() == "ok"


# --- Retry on transient failure ---


def test_firestore_retry_retries_then_succeeds():
    call_count = 0

    @firestore_retry
    def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("transient")
        return "recovered"

    assert flaky() == "recovered"
    assert call_count == 3


def test_llm_retry_retries_then_succeeds():
    call_count = 0

    @llm_retry
    def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise TimeoutError("503")
        return "recovered"

    assert flaky() == "recovered"
    assert call_count == 2


def test_embedding_retry_retries_then_succeeds():
    call_count = 0

    @embedding_retry
    def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RuntimeError("rate limit")
        return "recovered"

    assert flaky() == "recovered"
    assert call_count == 2


# --- Exhausted retries reraise ---


def test_firestore_retry_exhausted_reraises():
    @firestore_retry
    def always_fail():
        raise ConnectionError("permanent")

    with pytest.raises(ConnectionError, match="permanent"):
        always_fail()


def test_llm_retry_exhausted_reraises():
    @llm_retry
    def always_fail():
        raise TimeoutError("permanent")

    with pytest.raises(TimeoutError, match="permanent"):
        always_fail()


# --- Async support ---


@pytest.mark.asyncio
async def test_firestore_retry_async():
    call_count = 0

    @firestore_retry
    async def flaky_async():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("transient")
        return "recovered"

    result = await flaky_async()
    assert result == "recovered"
    assert call_count == 3


# --- firestore_write_with_retry ---


@pytest.mark.asyncio
async def test_firestore_write_with_retry_succeeds():
    doc_ref = AsyncMock()
    doc_ref.set = AsyncMock()

    await firestore_write_with_retry(doc_ref, {"key": "val"}, merge=True)

    doc_ref.set.assert_awaited_once_with({"key": "val"}, merge=True)


@pytest.mark.asyncio
async def test_firestore_write_with_retry_retries_on_failure():
    doc_ref = AsyncMock()
    call_count = 0

    async def flaky_set(data, merge=False):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("transient")

    doc_ref.set = flaky_set

    await firestore_write_with_retry(doc_ref, {"key": "val"})
    assert call_count == 3


@pytest.mark.asyncio
async def test_firestore_write_with_retry_exhausted_reraises():
    doc_ref = AsyncMock()
    doc_ref.set = AsyncMock(side_effect=ConnectionError("permanent"))

    with pytest.raises(ConnectionError, match="permanent"):
        await firestore_write_with_retry(doc_ref, {"key": "val"})

    assert doc_ref.set.await_count == 3
