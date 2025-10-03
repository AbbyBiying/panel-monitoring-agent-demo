# panel_monitoring/panel_agent_vertexai.py
# LangGraph Agent using Vertex AI (Gemini) via ChatVertexAI

import logging
import os
from datetime import datetime, timezone

from dotenv import find_dotenv, load_dotenv
from langchain_google_vertexai import ChatVertexAI, HarmBlockThreshold, HarmCategory
from langsmith import Client
from pydantic import ValidationError

from panel_monitoring.app.graph import build_graph
from panel_monitoring.app.runtime import run_interactive
from panel_monitoring.app.utils import (
    LLMClassificationError,
    automated_event_signals,
    build_classify_messages,
    get_event_input,
    load_credentials,
    looks_like_automated,
)
from panel_monitoring.app.schemas import Signals

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

ENV = os.getenv("ENV", "dev").lower()
load_dotenv(find_dotenv(), override=True)

PROJECT = os.getenv("LANGSMITH_PROJECT", "panel-monitoring-agent")
SESSION = (
    os.getenv("LANGSMITH_SESSION") or f"run-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# Use a widely available Vertex model by default (no pinned revisions like -002)
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

try:
    creds = load_credentials()
    llm = ChatVertexAI(
        model_name=DEFAULT_MODEL,
        temperature=0,
        project=PROJECT_ID,
        location=LOCATION,
        safety_settings=SAFETY_SETTINGS,
        credentials=creds,
    )
    logger.info(
        f"Vertex AI LLM ({DEFAULT_MODEL}) initialized in {PROJECT_ID}/{LOCATION}."
    )
except Exception as e:
    logger.error(f"Failed to initialize ChatVertexAI: {e}")
    llm = None


def _classify_with_vertexai(event: str) -> dict:
    """Classify an event into a Signals-shaped dict using Vertex AI."""
    if looks_like_automated(event):
        return automated_event_signals()
    if llm is None:
        raise LLMClassificationError(
            "Vertex AI client is not initialized. Check ADC/project/region."
        )
    try:
        structured_llm = llm.with_structured_output(Signals, include_raw=False)
        signals_obj = structured_llm.invoke(build_classify_messages(event))
        signals = signals_obj.model_dump()
        usage = getattr(signals_obj, "usage_metadata", None) or {}
        signals["_usage"] = usage

        if signals.get("suspicious_signup") == signals.get("normal_signup"):
            raise LLMClassificationError(
                f"Ambiguous classification from LLM: {signals}"
            )
        return signals
    except ValidationError as ve:
        raise LLMClassificationError(
            f"Validation error in Signals schema: {ve}"
        ) from ve
    except Exception as e:
        raise LLMClassificationError(
            f"Vertex classification error: {type(e).__name__}: {e}"
        ) from e


def graph():
    """Build the LangGraph workflow with Vertex classifier as provider."""
    return build_graph(_classify_with_vertexai)


def main():
    try:
        Client().create_project(PROJECT, upsert=True)
        logger.info(f"Agent ready. LangSmith Project: {PROJECT}")
    except Exception:
        logger.warning("Agent ready. (LangSmith tracing disabled)")

    app = graph()
    run_interactive(app, get_event_input=get_event_input, project_name=PROJECT)


if __name__ == "__main__":
    main()
