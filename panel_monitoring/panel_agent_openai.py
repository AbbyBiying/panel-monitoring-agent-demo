# panel_monitoring/panel_agent_openai.py
# LangGraph Agent using OpenAI via ChatOpenAI

import logging
import os
from datetime import datetime, timezone

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
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

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

try:
    llm = ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=0,
        timeout=30,
        max_retries=2,
        seed=0,
    )
    logger.info(f"OpenAI LLM ({DEFAULT_MODEL}) initialized via API Key.")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI LLM: {e}")
    llm = None


def _classify_with_openai(event: str) -> dict:
    """Classify an event into a Signals-shaped dict using OpenAI."""
    api_key = os.getenv("OPENAI_API_KEY")

    if looks_like_automated(event):
        return automated_event_signals()

    if llm is None or api_key is None:
        raise LLMClassificationError("OpenAI client is not initialized.")

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
            f"OpenAI classification error: {type(e).__name__}: {e}"
        ) from e


def graph():
    """Build the LangGraph workflow with OpenAI classifier as provider."""
    return build_graph(_classify_with_openai)


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
