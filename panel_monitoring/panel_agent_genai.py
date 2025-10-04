# panel_monitoring/panel_agent_genai.py
# LangGraph Agent using Google GenAI (Gemini API Key) via ChatGoogleGenerativeAI

import logging
import os
from datetime import datetime, timezone

from dotenv import find_dotenv, load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langsmith import Client
from pydantic import ValidationError

from panel_monitoring.app.graph import build_graph
from panel_monitoring.app.runtime import run_interactive
from panel_monitoring.app.utils import (
    LLMClassificationError,
    automated_event_signals,
    build_classify_messages,
    get_event_input,
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

os.environ.setdefault("LANGSMITH_PROJECT", PROJECT)

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "Set GOOGLE_API_KEY in your environment for ChatGoogleGenerativeAI."
    )

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

try:
    llm = ChatGoogleGenerativeAI(
        model=DEFAULT_MODEL,
        temperature=0,
        safety_settings=SAFETY_SETTINGS,
        google_api_key=GOOGLE_API_KEY,
    )
    logger.info(f"Google GenAI LLM ({DEFAULT_MODEL}) initialized via API Key.")
except Exception as e:
    logger.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}")
    llm = None


def _classify_with_genai(event: str) -> dict:
    """Classify an event into a Signals-shaped dict using Google GenAI."""
    if looks_like_automated(event):
        return automated_event_signals()

    if llm is None:
        raise LLMClassificationError("Google GenAI client is not initialized.")

    try:
        structured_llm = llm.with_structured_output(Signals, include_raw=False)
        signals_obj = structured_llm.invoke(build_classify_messages(event))
        signals = signals_obj.model_dump()

        if signals.get("suspicious_signup") == signals.get("normal_signup"):
            raise LLMClassificationError(f"Ambiguous classification: {signals}")

        return signals

    except ValidationError as ve:
        raise LLMClassificationError(f"Validation error: {ve}") from ve
    except Exception as e:
        raise LLMClassificationError(
            f"GenAI classification error: {type(e).__name__}: {e}"
        ) from e


def graph():
    """Build the LangGraph workflow with Google GenAI classifier as provider."""
    return build_graph(_classify_with_genai)


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
