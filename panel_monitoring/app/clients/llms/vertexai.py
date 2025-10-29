# panel_monitoring/app/clients/llms/vertexai.py

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from google.oauth2 import service_account
from dotenv import load_dotenv, find_dotenv
from google.api_core.exceptions import GoogleAPIError, NotFound, PermissionDenied
from pydantic import ValidationError
from langchain_google_vertexai import (
    ChatVertexAI,
    HarmBlockThreshold,
    HarmCategory,
)

from panel_monitoring.app.clients.llms.base import (
    LLMPredictionClient,
    PredictionError,
    PredictionResult,
)
from panel_monitoring.app.config import get_settings
from panel_monitoring.app.schemas import Signals
from panel_monitoring.app.prompts import PROMPT_CLASSIFY_SYSTEM, PROMPT_CLASSIFY_USER
from panel_monitoring.app.utils import (
    build_classify_messages,
    load_credentials,
    normalize_signals,
    parse_signals_from_text,
)

logger = logging.getLogger(__name__)

# Keep your env bootstrap behavior
load_dotenv(find_dotenv(), override=True)
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

DEFAULT_MODEL = "gemini-2.5-pro"

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}


SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def make_credentials_from_env():
    raw = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not raw:
        return None  # fall back to ADC

    # If it's a JSON path on disk
    if not raw.strip().startswith("{"):
        return service_account.Credentials.from_service_account_file(raw, scopes=SCOPES)

    # If it's a JSON string
    info = json.loads(raw)
    return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)


class LLMClientVertexAI(LLMPredictionClient):
    """
    Vertex AI (Gemini) client that returns Signals-shaped dicts via structured output,
    with a fallback to raw-text JSON parsing if needed.
    """

    def __init__(
        self,
        *,
        model_ref: str = "vertexai-classifier",
        model: str = DEFAULT_MODEL,
        user_prompt: str = PROMPT_CLASSIFY_USER,  # must contain "{event}"
        system_prompt: Optional[str] = PROMPT_CLASSIFY_SYSTEM,
        prompt_config: Optional[dict] = None,
        log=None,
    ):
        super().__init__(
            model_ref=model_ref,
            model_name=model,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            prompt_config=prompt_config,
            log=log,
        )
        self.model: Optional[str] = model
        self.project: Optional[str] = None
        self.location: Optional[str] = None
        self.client: Optional[ChatVertexAI] = None

    # ---- lifecycle --------------------------------------------------------

    def setup(self) -> None:
        """
        Initialize ChatVertexAI client with creds + safety.
        Project/Location precedence:
          settings -> env GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION
        """
        st = get_settings()

        self.project = st.google_cloud_project or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project:
            raise PredictionError("GOOGLE_CLOUD_PROJECT missing", str(self.model_ref))

        self.location = st.google_cloud_location or os.getenv(
            "GOOGLE_CLOUD_LOCATION", "us-central1"
        )

        model = self.model or DEFAULT_MODEL
        temperature = (self.prompt_config or {}).get("temperature", 0)
        max_retries = (self.prompt_config or {}).get("max_retries", 2)

        # if os.getenv("LG_GRAPH_NAME"):
        #     creds = json.loads(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        # else:
        #     creds = load_credentials()
        if os.getenv("ENVIRONMENT") == "local":
            logger.info("Running in local environment, loading credentials from file.")
            creds = load_credentials()
        else:
            creds = make_credentials_from_env()

            logger.info(
                "Running in NOT local environment, loading credentials from Path."
            )

        logger.debug("creds type: %s", type(creds))
        logger.info(
            "VertexAI config: project=%s location=%s model=%s",
            self.project or "<auto>",
            self.location,
            model,
        )

        self.client = ChatVertexAI(
            model=model,
            temperature=temperature,
            project=self.project,
            location=self.location,
            safety_settings=SAFETY_SETTINGS,
            credentials=creds,
            max_retries=max_retries,
        )

    # ---- primary (structured) APIs ---------------------------------------

    def classify_event(self, event: str) -> dict:
        """
        Sync structured classification with raw-text fallback.
        """
        if self.client is None:
            raise PredictionError(
                "Model not initialized. Call setup() first.", str(self.model_ref)
            )

        msgs = build_classify_messages(event)

        # Structured output first
        try:
            result = self.client.with_structured_output(Signals).invoke(msgs)
            return normalize_signals(result)
        except (ValidationError, NotFound, PermissionDenied, GoogleAPIError) as e:
            logger.debug(
                "Structured failed (%s): %s — falling back to raw JSON parse",
                type(e).__name__,
                e,
            )
        except Exception as e:
            logger.debug(
                "Structured failed (unexpected %s): %s — falling back",
                type(e).__name__,
                e,
                exc_info=True,
            )

        try:
            raw_resp = self.client.invoke(msgs)
            raw_text = getattr(raw_resp, "content", None) or str(raw_resp)
            return parse_signals_from_text(raw_text)
        except (ValidationError, NotFound, PermissionDenied, GoogleAPIError) as e:
            logger.debug(
                "Fallback raw parse failed (%s): %s", type(e).__name__, e, exc_info=True
            )
            raise PredictionError(
                f"Vertex AI classification error in fallback: {type(e).__name__}: {e}",
                str(self.model_ref),
            ) from e
        except Exception as e2:
            raise PredictionError(
                f"Vertex AI classification error after fallback: {type(e2).__name__}: {e2}",
                str(self.model_ref),
            ) from e2

    async def aclassify_event(self, event: str) -> dict:
        """
        Async structured classification with raw-text fallback.
        """
        if self.client is None:
            raise PredictionError(
                "Model not initialized. Call setup() first.", str(self.model_ref)
            )

        msgs = build_classify_messages(event)

        try:
            result = await self.client.with_structured_output(Signals).ainvoke(msgs)
            return normalize_signals(result)
        except (ValidationError, NotFound, PermissionDenied, GoogleAPIError) as e:
            logger.debug(
                "Structured output failed (%s): %s — falling back to raw parse",
                type(e).__name__,
                e,
            )
        except Exception as e:
            logger.debug(
                "Structured output failed (unexpected %s): %s — falling back",
                type(e).__name__,
                e,
                exc_info=True,
            )

        try:
            raw_resp = await self.client.ainvoke(msgs)  # ✅
            raw_text = getattr(raw_resp, "content", None) or str(raw_resp)
            return parse_signals_from_text(raw_text)
        except (ValidationError, NotFound, PermissionDenied, GoogleAPIError) as e:
            logger.debug(
                "Fallback raw parse failed (%s): %s", type(e).__name__, e, exc_info=True
            )
            raise PredictionError(
                f"Vertex AI classification error in fallback: {type(e).__name__}: {e}",
                str(self.model_ref),
            ) from e
        except Exception as e2:
            raise PredictionError(
                f"Vertex AI classification error after fallback: {type(e2).__name__}: {e2}",
                str(self.model_ref),
            ) from e2

    # ---- base.py-required async predict ----------------------------------

    async def predict(self, prompt: str) -> PredictionResult:
        """
        Conforms to minimal base: treat `prompt` as event text, run classifier,
        return PredictionResult with JSON of normalized Signals.
        """
        start = time.perf_counter()
        try:
            data = await self.aclassify_event(prompt)
            text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        except PredictionError:
            logger.debug("Re-raising PredictionError")
            raise
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000.0
            raise PredictionError(
                f"Vertex AI predict failed: {type(e).__name__}: {e}",
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

_client_singleton: Optional[LLMClientVertexAI] = None


def classify_with_vertexai(event: str) -> dict:
    """
    Back-compat wrapper so old call sites still work.
    """
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = LLMClientVertexAI()
        _client_singleton.setup()
    return _client_singleton.classify_event(event)
