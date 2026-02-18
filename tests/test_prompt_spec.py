# tests/test_prompt_spec.py
"""Unit tests for the PromptSpec → Firestore integration."""
from __future__ import annotations

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("LANGSMITH_TRACING", "false")

from panel_monitoring.models.firestore_docs import PromptSpecDoc
from panel_monitoring.app.utils import build_classify_messages
from panel_monitoring.app.prompts import PROMPT_CLASSIFY_SYSTEM


# ---------------------------------------------------------------------------
# 1. PromptSpecDoc model
# ---------------------------------------------------------------------------


class TestPromptSpecDoc:
    def test_defaults(self):
        doc = PromptSpecDoc()
        assert doc.deployment_status == "deactivated"
        assert doc.deployment_role is None
        assert doc.model_name == ""
        assert doc.system_prompt == ""
        assert doc.user_prompt == ""
        assert doc.config == {}
        assert doc.labels == []
        assert doc.url is None
        assert doc.doc_id is None

    def test_full_construction(self):
        doc = PromptSpecDoc(
            model_host="vertexai",
            model_name="gemini-2.5-flash",
            system_prompt="You are a classifier.",
            prompt="Classify: {event}",
            config={"temperature": 0},
            version="3",
            labels=["fraud"],
            url="https://custom.endpoint/v1",
            deployment_status="live",
            deployment_role="signup_classification",
        )
        assert doc.deployment_status == "live"
        assert doc.deployment_role == "signup_classification"
        assert doc.version == "3"
        assert doc.url == "https://custom.endpoint/v1"

    def test_doc_id_excluded_from_dump(self):
        doc = PromptSpecDoc()
        doc.doc_id = "abc123"
        dumped = doc.model_dump()
        assert "doc_id" not in dumped
        assert doc.doc_id == "abc123"

    def test_invalid_deployment_status_rejected(self):
        with pytest.raises(Exception):
            PromptSpecDoc(deployment_status="invalid_status")

    def test_all_deployment_statuses(self):
        for status in ("deactivated", "pre_live", "canary", "live", "failover"):
            doc = PromptSpecDoc(deployment_status=status)
            assert doc.deployment_status == status


# ---------------------------------------------------------------------------
# 2. get_active_prompt_spec (mocked Firestore)
# ---------------------------------------------------------------------------


def _make_fake_doc_snapshot(doc_id: str, data: dict):
    """Create a mock Firestore document snapshot."""
    snap = MagicMock()
    snap.id = doc_id
    snap.to_dict.return_value = data
    return snap


def _make_fake_col_and_query(snapshots: list):
    """
    Build a mock collection + chained query that matches the FieldFilter calling
    convention:  col.where(filter=...).where(filter=...).limit(1).stream()

    Returns (fake_col, fake_query) so callers can inspect .where() calls.
    """
    async def fake_stream():
        for s in snapshots:
            yield s

    fake_query = MagicMock()
    fake_query.where.return_value = fake_query
    fake_query.limit.return_value = fake_query
    fake_query.stream.return_value = fake_stream()

    fake_col = MagicMock()
    fake_col.where.return_value = fake_query

    return fake_col, fake_query


class TestGetActivePromptSpec:
    @pytest.mark.asyncio
    async def test_returns_spec_when_live_doc_exists(self):
        fake_data = {
            "model_host": "vertexai",
            "model_name": "gemini-2.5-flash",
            "system_prompt": "sys",
            "user_prompt": "usr {event}",
            "config": {},
            "version": "1",
            "labels": [],
            "deployment_status": "live",
            "deployment_role": "signup_classification",
        }
        fake_snap = _make_fake_doc_snapshot("prompt_abc", fake_data)
        fake_col, _ = _make_fake_col_and_query([fake_snap])

        with patch(
            "panel_monitoring.data.firestore_client.prompt_specs_col",
            new_callable=AsyncMock,
            return_value=fake_col,
        ):
            from panel_monitoring.data.firestore_client import get_active_prompt_spec

            result = await get_active_prompt_spec("signup_classification")

        assert result is not None
        assert isinstance(result, PromptSpecDoc)
        assert result.doc_id == "prompt_abc"
        assert result.system_prompt == "sys"
        assert result.deployment_status == "live"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_live_doc(self):
        fake_col, _ = _make_fake_col_and_query([])

        with patch(
            "panel_monitoring.data.firestore_client.prompt_specs_col",
            new_callable=AsyncMock,
            return_value=fake_col,
        ):
            from panel_monitoring.data.firestore_client import get_active_prompt_spec

            result = await get_active_prompt_spec("signup_classification")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        with patch(
            "panel_monitoring.data.firestore_client.prompt_specs_col",
            new_callable=AsyncMock,
            side_effect=RuntimeError("connection failed"),
        ):
            from panel_monitoring.data.firestore_client import get_active_prompt_spec

            result = await get_active_prompt_spec("signup_classification")

        assert result is None

    @pytest.mark.asyncio
    async def test_queries_use_field_filter_keyword_arg(self):
        """Verify .where() is called with filter= keyword to avoid deprecation warning."""
        from google.cloud.firestore_v1.base_query import FieldFilter

        fake_col, fake_query = _make_fake_col_and_query([])

        with patch(
            "panel_monitoring.data.firestore_client.prompt_specs_col",
            new_callable=AsyncMock,
            return_value=fake_col,
        ):
            from panel_monitoring.data.firestore_client import get_active_prompt_spec

            await get_active_prompt_spec("signup_classification")

        # First .where() called on the collection itself
        col_where_call = fake_col.where.call_args
        assert "filter" in col_where_call.kwargs
        f1 = col_where_call.kwargs["filter"]
        assert isinstance(f1, FieldFilter)
        assert f1.field_path == "deployment_status"
        assert f1.value == "live"

        # Second .where() called on the chained query
        query_where_call = fake_query.where.call_args
        assert "filter" in query_where_call.kwargs
        f2 = query_where_call.kwargs["filter"]
        assert isinstance(f2, FieldFilter)
        assert f2.field_path == "deployment_role"
        assert f2.value == "signup_classification"

    @pytest.mark.asyncio
    async def test_ignores_wrong_role(self):
        """Even if a doc is returned, the query itself filters by role — verify the
        query receives the role we pass in."""
        fake_col, fake_query = _make_fake_col_and_query([])

        with patch(
            "panel_monitoring.data.firestore_client.prompt_specs_col",
            new_callable=AsyncMock,
            return_value=fake_col,
        ):
            from panel_monitoring.data.firestore_client import get_active_prompt_spec

            await get_active_prompt_spec("some_other_role")

        query_where_call = fake_query.where.call_args
        f = query_where_call.kwargs["filter"]
        assert f.value == "some_other_role"


# ---------------------------------------------------------------------------
# 3. build_classify_messages with overrides
# ---------------------------------------------------------------------------


class TestBuildClassifyMessagesOverrides:
    def test_uses_hardcoded_by_default(self):
        msgs = build_classify_messages("test event")
        assert msgs[0].content == PROMPT_CLASSIFY_SYSTEM
        assert "test event" in msgs[1].content

    def test_system_prompt_override(self):
        custom_sys = "Custom system prompt."
        msgs = build_classify_messages("test event", system_prompt_override=custom_sys)
        assert msgs[0].content == custom_sys
        assert "test event" in msgs[1].content

    def test_user_prompt_override(self):
        custom_usr = "Custom user: {event}"
        msgs = build_classify_messages("test event", user_prompt_override=custom_usr)
        assert msgs[0].content == PROMPT_CLASSIFY_SYSTEM
        assert msgs[1].content == "Custom user: test event"

    def test_both_overrides(self):
        custom_sys = "SYS"
        custom_usr = "USR: {event}"
        msgs = build_classify_messages(
            "data",
            system_prompt_override=custom_sys,
            user_prompt_override=custom_usr,
        )
        assert msgs[0].content == "SYS"
        assert msgs[1].content == "USR: data"

    def test_none_overrides_fall_back(self):
        msgs = build_classify_messages(
            "data",
            system_prompt_override=None,
            user_prompt_override=None,
        )
        assert msgs[0].content == PROMPT_CLASSIFY_SYSTEM


# ---------------------------------------------------------------------------
# 4. signal_evaluation_node — prompt spec loading & fallback
# ---------------------------------------------------------------------------


def _make_state(**overrides):
    """Build a minimal GraphState-like object for node tests."""
    from panel_monitoring.app.schemas import GraphState

    defaults = {
        "event_data": {"identity": {"panelist_id": "test123"}},
        "event_text": '{"identity": {"panelist_id": "test123"}}',
        "event_id": "evt_001",
        "panelist_id": "test123",
        "retrieved_docs": [],
        "explanation_report": [],
    }
    defaults.update(overrides)
    return GraphState(**defaults)


class TestSignalEvaluationNodePromptSpec:
    @pytest.mark.asyncio
    async def test_uses_prompt_spec_when_available(self):
        spec = PromptSpecDoc(
            model_name="gemini-2.5-flash",
            system_prompt="Custom sys prompt.",
            user_prompt="Custom user: {event}",
            version="2",
            deployment_status="live",
            deployment_role="signup_classification",
        )
        spec.doc_id = "ps_123"

        fake_signals = (
            {
                "suspicious_signup": False,
                "normal_signup": True,
                "confidence": 0.85,
                "reason": "looks fine",
            },
            {"provider": "vertexai", "model": "gemini-2.5-flash"},
        )

        state = _make_state()

        with (
            patch(
                "panel_monitoring.app.nodes.get_active_prompt_spec",
                new_callable=AsyncMock,
                return_value=spec,
            ),
            patch(
                "panel_monitoring.app.nodes.aclassify_event",
                new_callable=AsyncMock,
                return_value=fake_signals,
            ) as mock_classify,
        ):
            from panel_monitoring.app.nodes import signal_evaluation_node

            result = await signal_evaluation_node(state)

        # Verify overrides were passed
        mock_classify.assert_awaited_once()
        call_kwargs = mock_classify.call_args
        assert call_kwargs.kwargs["system_prompt_override"] == "Custom sys prompt."
        assert call_kwargs.kwargs["user_prompt_override"] == "Custom user: {event}"  # noqa: E501

        # Verify prompt_id/prompt_name propagated
        assert result["prompt_id"] == "ps_123"
        assert result["prompt_name"] == "signup_classification@v2"

    @pytest.mark.asyncio
    async def test_falls_back_when_no_prompt_spec(self):
        fake_signals = (
            {
                "suspicious_signup": False,
                "normal_signup": True,
                "confidence": 0.80,
                "reason": "ok",
            },
            {"provider": "vertexai", "model": "gemini-2.5-flash"},
        )

        state = _make_state()

        with (
            patch(
                "panel_monitoring.app.nodes.get_active_prompt_spec",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "panel_monitoring.app.nodes.aclassify_event",
                new_callable=AsyncMock,
                return_value=fake_signals,
            ) as mock_classify,
        ):
            from panel_monitoring.app.nodes import signal_evaluation_node

            result = await signal_evaluation_node(state)

        call_kwargs = mock_classify.call_args
        assert call_kwargs.kwargs["system_prompt_override"] is None
        assert call_kwargs.kwargs["user_prompt_override"] is None

        assert "prompt_id" not in result
        assert "prompt_name" not in result


# ---------------------------------------------------------------------------
# 5. save_classification_node — prompt_id/prompt_name in Firestore docs
# ---------------------------------------------------------------------------


class TestSaveClassificationNodePromptFields:
    def _mock_col(self):
        """Build a mock collection where .document(id) returns a mock with async .set()."""
        mock_doc = MagicMock()
        mock_doc.set = AsyncMock()
        mock_col = MagicMock()
        mock_col.document.return_value = mock_doc
        return mock_col, mock_doc

    @pytest.mark.asyncio
    async def test_prompt_fields_written_to_firestore(self):
        from panel_monitoring.app.schemas import ModelMeta

        state = _make_state(
            classification="normal",
            confidence=0.9,
            prompt_id="ps_456",
            prompt_name="signup_classification@v3",
            model_meta=ModelMeta(provider="vertexai", model="gemini-2.5-flash"),
        )
        state.signals = MagicMock()
        state.signals.model_dump.return_value = {
            "suspicious_signup": False,
            "normal_signup": True,
            "confidence": 0.9,
            "reason": "ok",
        }

        runs_col_obj, mock_run_doc = self._mock_col()
        events_col_obj, mock_event_doc = self._mock_col()

        with (
            patch(
                "panel_monitoring.app.nodes.runs_col",
                new_callable=AsyncMock,
                return_value=runs_col_obj,
            ),
            patch(
                "panel_monitoring.app.nodes.events_col",
                new_callable=AsyncMock,
                return_value=events_col_obj,
            ),
        ):
            from panel_monitoring.app.nodes import save_classification_node

            await save_classification_node(state)

        # Check runs doc
        run_set_call = mock_run_doc.set.call_args[0][0]
        assert run_set_call["prompt_id"] == "ps_456"
        assert run_set_call["prompt_name"] == "signup_classification@v3"

        # Check events doc
        event_set_call = mock_event_doc.set.call_args[0][0]
        assert event_set_call["prompt_id"] == "ps_456"
        assert event_set_call["prompt_name"] == "signup_classification@v3"

    @pytest.mark.asyncio
    async def test_prompt_fields_omitted_when_none(self):
        from panel_monitoring.app.schemas import ModelMeta

        state = _make_state(
            classification="normal",
            confidence=0.9,
            prompt_id=None,
            prompt_name=None,
            model_meta=ModelMeta(provider="vertexai", model="gemini-2.5-flash"),
        )
        state.signals = MagicMock()
        state.signals.model_dump.return_value = {
            "suspicious_signup": False,
            "normal_signup": True,
            "confidence": 0.9,
            "reason": "ok",
        }

        runs_col_obj, mock_run_doc = self._mock_col()
        events_col_obj, mock_event_doc = self._mock_col()

        with (
            patch(
                "panel_monitoring.app.nodes.runs_col",
                new_callable=AsyncMock,
                return_value=runs_col_obj,
            ),
            patch(
                "panel_monitoring.app.nodes.events_col",
                new_callable=AsyncMock,
                return_value=events_col_obj,
            ),
        ):
            from panel_monitoring.app.nodes import save_classification_node

            await save_classification_node(state)

        run_set_call = mock_run_doc.set.call_args[0][0]
        assert "prompt_id" not in run_set_call
        assert "prompt_name" not in run_set_call

        event_set_call = mock_event_doc.set.call_args[0][0]
        assert "prompt_id" not in event_set_call
        assert "prompt_name" not in event_set_call


# ---------------------------------------------------------------------------
# 6. prompt_specs_col — collection name
# ---------------------------------------------------------------------------


class TestPromptSpecsCol:
    @pytest.mark.asyncio
    async def test_returns_prompt_specs_collection(self):
        mock_db = MagicMock()
        mock_db.collection.return_value = MagicMock()

        with patch(
            "panel_monitoring.data.firestore_client.get_db",
            new_callable=AsyncMock,
            return_value=mock_db,
        ):
            from panel_monitoring.data.firestore_client import prompt_specs_col

            await prompt_specs_col()

        mock_db.collection.assert_called_once_with("prompt_specs")


# ---------------------------------------------------------------------------
# 7. Push script — versioning and immutability
# ---------------------------------------------------------------------------


def _make_push_col(existing_docs: list[dict]):
    """
    Build a mock collection for push_prompt_to_firestore tests.

    existing_docs: list of dicts representing existing Firestore docs
                   (each must have a 'version' key).
    """
    async def fake_stream():
        for data in existing_docs:
            snap = MagicMock()
            snap.to_dict.return_value = data
            yield snap

    fake_query = MagicMock()
    fake_query.stream.return_value = fake_stream()

    captured = {}

    async def capture_set(data):
        captured.update(data)

    mock_doc = MagicMock()
    mock_doc.set = capture_set

    mock_col = MagicMock()
    mock_col.where.return_value = fake_query
    mock_col.document.return_value = mock_doc

    return mock_col, captured, mock_col.document


class TestPushPromptToFirestore:
    @pytest.mark.asyncio
    async def test_first_push_creates_v1(self):
        """When no existing docs, push creates signup_classification_v1."""
        mock_col, captured, mock_document = _make_push_col(existing_docs=[])

        with patch(
            "panel_monitoring.scripts.push_prompt_to_firestore.prompt_specs_col",
            new_callable=AsyncMock,
            return_value=mock_col,
        ):
            from panel_monitoring.scripts.push_prompt_to_firestore import push
            await push()

        mock_document.assert_called_once_with("signup_classification_v1")
        assert captured["version"] == "1"
        assert captured["deployment_role"] == "signup_classification"
        assert captured["deployment_status"] == "pre_live"

    @pytest.mark.asyncio
    async def test_second_push_creates_v2(self):
        """When v1 already exists, push creates v2."""
        mock_col, captured, mock_document = _make_push_col(
            existing_docs=[{"version": "1", "deployment_role": "signup_classification"}]
        )

        with patch(
            "panel_monitoring.scripts.push_prompt_to_firestore.prompt_specs_col",
            new_callable=AsyncMock,
            return_value=mock_col,
        ):
            from panel_monitoring.scripts.push_prompt_to_firestore import push
            await push()

        mock_document.assert_called_once_with("signup_classification_v2")
        assert captured["version"] == "2"

    @pytest.mark.asyncio
    async def test_push_sets_correct_prompts(self):
        """Pushed doc contains the actual system and user prompts."""
        from panel_monitoring.app.prompts import PROMPT_CLASSIFY_SYSTEM, PROMPT_CLASSIFY_USER

        mock_col, captured, _ = _make_push_col(existing_docs=[])

        with patch(
            "panel_monitoring.scripts.push_prompt_to_firestore.prompt_specs_col",
            new_callable=AsyncMock,
            return_value=mock_col,
        ):
            from panel_monitoring.scripts.push_prompt_to_firestore import push
            await push()

        assert captured["system_prompt"] == PROMPT_CLASSIFY_SYSTEM
        assert captured["user_prompt"] == PROMPT_CLASSIFY_USER

    @pytest.mark.asyncio
    async def test_push_never_uses_merge(self):
        """Push must call set() without merge=True to enforce immutability."""
        mock_col, _, _ = _make_push_col(existing_docs=[])

        set_calls = []

        async def capture_set(data, **kwargs):
            set_calls.append(kwargs)

        mock_col.document.return_value.set = capture_set

        with patch(
            "panel_monitoring.scripts.push_prompt_to_firestore.prompt_specs_col",
            new_callable=AsyncMock,
            return_value=mock_col,
        ):
            from panel_monitoring.scripts.push_prompt_to_firestore import push
            await push()

        assert len(set_calls) == 1
        assert set_calls[0].get("merge") is not True, "push must NOT use merge=True"

    @pytest.mark.asyncio
    async def test_push_starts_as_pre_live_not_live(self):
        """New versions must never be pushed directly to live."""
        mock_col, captured, _ = _make_push_col(existing_docs=[])

        with patch(
            "panel_monitoring.scripts.push_prompt_to_firestore.prompt_specs_col",
            new_callable=AsyncMock,
            return_value=mock_col,
        ):
            from panel_monitoring.scripts.push_prompt_to_firestore import push
            await push()

        assert captured["deployment_status"] == "pre_live"
        assert captured["deployment_status"] != "live"


# ---------------------------------------------------------------------------
# 8. Seed script — validate seeded document shape
# ---------------------------------------------------------------------------


class TestSeedFirestorePromptSpec:
    @pytest.mark.asyncio
    async def test_seed_writes_valid_prompt_spec(self):
        """Verify run_seed() writes a prompt_specs doc with the required fields."""
        captured_data = {}

        mock_ps_doc = MagicMock()
        mock_ps_doc.id = "signup_classification_v1"

        async def capture_set(data, **kwargs):
            captured_data.update(data)

        mock_ps_doc.set = capture_set

        mock_ps_col = MagicMock()
        mock_ps_col.document.return_value = mock_ps_doc

        # Mock all col helpers so run_seed() doesn't hit real Firestore
        mock_generic_doc = MagicMock()
        mock_generic_doc.id = "mock_id"
        mock_generic_doc.set = AsyncMock()

        mock_generic_col = MagicMock()
        mock_generic_col.document.return_value = mock_generic_doc

        with (
            patch(
                "panel_monitoring.scripts.seed_firestore.projects_col",
                new_callable=AsyncMock,
                return_value=mock_generic_col,
            ),
            patch(
                "panel_monitoring.scripts.seed_firestore.events_col",
                new_callable=AsyncMock,
                return_value=mock_generic_col,
            ),
            patch(
                "panel_monitoring.scripts.seed_firestore.alerts_col",
                new_callable=AsyncMock,
                return_value=mock_generic_col,
            ),
            patch(
                "panel_monitoring.scripts.seed_firestore.prompt_specs_col",
                new_callable=AsyncMock,
                return_value=mock_ps_col,
            ),
        ):
            from panel_monitoring.scripts.seed_firestore import run_seed

            await run_seed()

        # The seeded doc should be parseable as a valid PromptSpecDoc
        assert captured_data["deployment_status"] == "live"
        assert captured_data["deployment_role"] == "signup_classification"
        assert captured_data["model_name"] == "gemini-2.5-flash"
        assert "{event}" in captured_data["user_prompt"]
        assert len(captured_data["system_prompt"]) > 0

        # Validate it round-trips through the Pydantic model
        filterable = {
            k: v
            for k, v in captured_data.items()
            if k not in ("created_at", "updated_at")  # SERVER_TIMESTAMP sentinels
        }
        doc = PromptSpecDoc.model_validate(filterable)
        assert doc.deployment_status == "live"
        assert doc.version == "1"
