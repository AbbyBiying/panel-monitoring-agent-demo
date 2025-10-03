# panel_monitoring/app/clients/llms/gemini.py

from __future__ import annotations

import json
import os
import time
from typing import Optional

from pydantic import ValidationError
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmCategory,
    HarmBlockThreshold,
)

from panel_monitoring.app.clients.llms.base import (
    LLMPredictionClient,
    PredictionError,
    PredictionResult,
)
from panel_monitoring.app.config import get_settings
from panel_monitoring.app.schemas import Signals
from panel_monitoring.app.prompts import PROMPT_CLASSIFY_SYSTEM, PROMPT_CLASSIFY_USER
from panel_monitoring.app.utils import build_classify_messages, normalize_signals


DEFAULT_MODEL = "gemini-2.5-pro"

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}


class LLMClientGemini(LLMPredictionClient):
    """
    Gemini client that produces structured `Signals` output (same behavior as the original function).
    """

    def __init__(
        self,
        *,
        model_ref: str = "gemini-classifier",
        model_name: str = DEFAULT_MODEL,
        user_prompt: str = PROMPT_CLASSIFY_USER,   # must contain "{event}"
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
        self.api_key: Optional[str] = None
        self.client: Optional[ChatGoogleGenerativeAI] = None


    def setup(self) -> None:
        """
        Initialize ChatGoogleGenerativeAI client with safety settings.
        API key resolution precedence:
          prompt_config["api_key"] -> get_settings().google_api_key -> env GOOGLE_API_KEY
        """
        settings = get_settings()
        self.api_key = (
            (self.prompt_config or {}).get("api_key")
            or getattr(settings, "google_api_key", None)
            or os.getenv("GOOGLE_API_KEY")
        )
        if not self.api_key:
            raise PredictionError("GOOGLE_API_KEY missing", str(self.model_ref))

        model_name = self.model_name or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        temperature = (self.prompt_config or {}).get("temperature", 0)
        timeout = (self.prompt_config or {}).get("timeout", 30)
        max_retries = (self.prompt_config or {}).get("max_retries", 2)

        self.client = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
            safety_settings=SAFETY_SETTINGS,
            google_api_key=self.api_key,
        )


    # ---- primary (structured) APIs ---------------------------------------

    def classify_event(self, event: str) -> dict:
        """
        Synchronous structured classification (mirrors original function).
        Returns a dict normalized to your Signals shape.
        """
        if self.client is None:
            raise PredictionError("Model not initialized. Call setup() first.", str(self.model_ref))

        msgs = build_classify_messages(event)
        try:
            result = self.client.with_structured_output(Signals).invoke(msgs)
            return normalize_signals(result)
        except ValidationError as e:
            raise PredictionError(f"Schema validation failed: {e}", str(self.model_ref)) from e
        except Exception as e:
            raise PredictionError(
                f"GenAI classification error: {type(e).__name__}: {e}",
                str(self.model_ref),
            ) from e

    async def aclassify_event(self, event: str) -> dict:
        """
        Async structured classification.
        """
        if self.client is None:
            raise PredictionError("Model not initialized. Call setup() first.", str(self.model_ref))

        msgs = build_classify_messages(event)
        try:
            result = await self.client.with_structured_output(Signals).ainvoke(msgs)
            return normalize_signals(result)
        except ValidationError as e:
            raise PredictionError(f"Schema validation failed: {e}", str(self.model_ref)) from e
        except Exception as e:
            raise PredictionError(
                f"GenAI classification error: {type(e).__name__}: {e}",
                str(self.model_ref),
            ) from e

    # ---- base.py-required async predict ----------------------------------

    async def predict(self, prompt: str) -> PredictionResult:
        """
        Conforms to the minimal base: treats `prompt` as the event text,
        runs the structured classifier, and returns a PredictionResult
        whose text is the JSON of normalized Signals.
        """
        start = time.perf_counter()
        try:
            data = await self.aclassify_event(prompt)
            text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        except PredictionError:
            # already annotated with duration in caller if needed
            raise
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000.0
            raise PredictionError(
                f"Gemini predict failed: {type(e).__name__}: {e}",
                str(self.model_ref),
                duration_ms,
            ) from e

        duration_ms = (time.perf_counter() - start) * 1000.0
        return PredictionResult(
            model_ref=str(self.model_ref),
            prompt=prompt,
            text=text,
            duration_ms=duration_ms,
        )


# ---- Back-compat convenience wrapper ---------------------------------------

# Keep the original function name so existing imports keep working
_client_singleton: Optional[LLMClientGemini] = None

def classify_with_genai(event: str) -> dict:
    """
    Back-compat thin wrapper that uses a singleton LLMClientGemini.
    """
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = LLMClientGemini()
        _client_singleton.setup()
    return _client_singleton.classify_event(event)
