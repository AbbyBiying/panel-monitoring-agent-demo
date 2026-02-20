# panel_monitoring/app/clients/llms/vertexai.py

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

from langchain_google_vertexai import ChatVertexAI, HarmBlockThreshold, HarmCategory
from panel_monitoring.app.clients.llms.base import LLMPredictionClient, PredictionResult
from panel_monitoring.app.config import get_settings
from panel_monitoring.app.schemas import Signals
from panel_monitoring.app.prompts import PROMPT_CLASSIFY_SYSTEM, PROMPT_CLASSIFY_USER
from panel_monitoring.app.retry import llm_retry
from panel_monitoring.app.utils import (
    build_classify_messages,
    load_credentials,
    make_credentials_from_env,
    normalize_signals,
    parse_signals_from_text,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}


class LLMClientVertexAI(LLMPredictionClient):
    def __init__(
        self,
        *,
        model_ref: str = "vertexai-classifier",
        model_name: str = os.getenv("VERTEX_MODEL", DEFAULT_MODEL),
        user_prompt: str = PROMPT_CLASSIFY_USER,
        system_prompt: Optional[str] = PROMPT_CLASSIFY_SYSTEM,
        prompt_config: Optional[dict] = None,
        log=None,
    ):
        super().__init__(
            model_ref=model_ref,
            model_name=model_name,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            prompt_config=prompt_config,
            log=log,
        )
        self.client: Optional[ChatVertexAI] = None

    def setup(self) -> None:
        """Performed at startup. Redundant load_dotenv removed."""
        st = get_settings()
        project = st.google_cloud_project or os.getenv("GOOGLE_CLOUD_PROJECT")
        location = st.google_cloud_location or os.getenv(
            "GOOGLE_CLOUD_LOCATION", "us-central1"
        )

        # Determine credentials once at setup
        if os.getenv("ENVIRONMENT") == "local":
            creds = load_credentials()
        else:
            creds = make_credentials_from_env()

        self.client = ChatVertexAI(
            model=self.model_name or DEFAULT_MODEL,
            temperature=0,
            project=project,
            location=location,
            safety_settings=SAFETY_SETTINGS,
            credentials=creds,
        )

    @llm_retry
    def _sync_classify(
        self,
        event: str,
        retrieved_docs: list[dict] | None = None,
        *,
        system_prompt_override: str | None = None,
        user_prompt_override: str | None = None,
    ) -> dict:
        """The core synchronous logic that might block on I/O."""
        if not self.client:
            raise RuntimeError("Client not setup.")

        msgs = build_classify_messages(
            event,
            retrieved_docs=retrieved_docs,
            system_prompt_override=system_prompt_override,
            user_prompt_override=user_prompt_override,
        )

        try:
            # Attempt structured output with raw response for metadata
            result = self.client.with_structured_output(Signals, include_raw=True).invoke(msgs)
            raw_msg = result["raw"]
            meta = {"usage": getattr(raw_msg, "usage_metadata", None) or {}}
            return normalize_signals(result["parsed"]), meta
        except Exception:
            # Fallback to raw text parsing
            raw_resp = self.client.invoke(msgs)
            raw_text = getattr(raw_resp, "content", "")
            meta = {"usage": getattr(raw_resp, "usage_metadata", None) or {}}
            return parse_signals_from_text(raw_text), meta

    async def aclassify_event(
        self,
        event: str,
        retrieved_docs: list[dict] | None = None,
        *,
        system_prompt_override: str | None = None,
        user_prompt_override: str | None = None,
    ) -> dict:
        """
        Async entry point. Offloads the blocking sync_classify
        to a separate thread pool.
        """
        # This keeps LangGraph's event loop completely free!
        return await asyncio.to_thread(
            self._sync_classify,
            event,
            retrieved_docs,
            system_prompt_override=system_prompt_override,
            user_prompt_override=user_prompt_override,
        )

    async def predict(self, prompt: str) -> PredictionResult:
        """Conforms to base.py async interface."""
        start = time.perf_counter()
        data = await self.aclassify_event(prompt)
        return PredictionResult(
            model_ref=self.model_ref,
            prompt=prompt,
            text=str(data),
            duration_ms=(time.perf_counter() - start) * 1000.0,
        )

    def classify_event(self, event: str) -> dict:
        """Sync classification (alias for _sync_classify for back-compat)."""
        return self._sync_classify(event)


# ---- Back-compat convenience wrapper ---------------------------------------

_client_singleton: Optional[LLMClientVertexAI] = None


def classify_with_vertexai(event: str) -> dict:
    """
    Back-compat sync wrapper for old call sites.
    For async usage, use aclassify_event() from the llms module instead.
    """
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = LLMClientVertexAI()
        _client_singleton.setup()
    return _client_singleton.classify_event(event)
