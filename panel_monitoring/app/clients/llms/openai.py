# panel_monitoring/app/clients/llms/openai.py

from __future__ import annotations

import json
import os
import time
from typing import Optional

from pydantic import ValidationError
from langchain_openai import ChatOpenAI

# OpenAI exception types (robust to package changes)
try:
    from openai import APIError, RateLimitError, AuthenticationError, OpenAIError  # type: ignore
except Exception:  # pragma: no cover
    APIError = RateLimitError = AuthenticationError = OpenAIError = Exception  # type: ignore

from panel_monitoring.app.clients.llms.base import (
    LLMPredictionClient,
    PredictionError,
    PredictionResult,
)
from panel_monitoring.app.config import get_settings
from panel_monitoring.app.schemas import Signals
from panel_monitoring.app.prompts import PROMPT_CLASSIFY_SYSTEM, PROMPT_CLASSIFY_USER
from panel_monitoring.app.utils import build_classify_messages, normalize_signals


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class LLMClientOpenAI(LLMPredictionClient):
    """
    OpenAI client that produces structured `Signals` output (same behavior as your original function).
    """

    def __init__(
        self,
        *,
        model_ref: str = "openai-classifier",
        model_name: str = DEFAULT_OPENAI_MODEL,
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
        self.client: Optional[ChatOpenAI] = None

    # ---- lifecycle --------------------------------------------------------

    def setup(self) -> None:
        """
        Initialize ChatOpenAI client.
        API key resolution precedence:
          prompt_config["api_key"] -> get_settings().openai_api_key -> env OPENAI_API_KEY
        """
        settings = get_settings()
        self.api_key = (
            (self.prompt_config or {}).get("api_key")
            or getattr(settings, "openai_api_key", None)
            or os.getenv("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise PredictionError("OPENAI_API_KEY missing", str(self.model_ref))

        model_name = self.model_name or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        temperature = (self.prompt_config or {}).get("temperature", 0)
        timeout = (self.prompt_config or {}).get("timeout", 30)
        max_retries = (self.prompt_config or {}).get("max_retries", 2)
        seed = (self.prompt_config or {}).get("seed", 0)

        self.client = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            seed=seed,
            api_key=self.api_key,
        )


    def classify_event(self, event: str) -> dict:
        """
        Synchronous structured classification (mirrors your original function).
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
                f"OpenAI classification error: {type(e).__name__}: {e}",
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
        except (AuthenticationError, RateLimitError, APIError, OpenAIError) as e:
            raise PredictionError(f"OpenAI API error: {e}", str(self.model_ref)) from e
        except Exception as e:
            raise PredictionError(
                f"Unexpected classification error: {type(e).__name__}: {e}",
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
            raise
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000.0
            raise PredictionError(
                f"OpenAI predict failed: {type(e).__name__}: {e}",
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

_client_singleton: Optional[LLMClientOpenAI] = None

def classify_with_openai(event: str) -> dict:
    """
    Back-compat thin wrapper so old call sites still work.
    """
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = LLMClientOpenAI()
        _client_singleton.setup()
    return _client_singleton.classify_event(event)
