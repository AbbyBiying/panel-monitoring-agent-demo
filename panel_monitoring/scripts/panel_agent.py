# panel_monitoring/scripts/panel_agent.py
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, Callable, Tuple

from langsmith import Client

from panel_monitoring.app.graph import build_graph
from panel_monitoring.app.runtime import run_interactive
from panel_monitoring.app.utils import get_event_input
from panel_monitoring.app.clients.llms import get_llm_classifier
from panel_monitoring.app.clients.llms.provider_base import (
    ClassifierProvider,
    FunctionProvider,
    UserEvent,
    SignalMeta,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("openai", "genai", "gemini", "vertexai")


def _model_from_obj_or_env(obj: Any, provider_key: str) -> str:
    """Try to get model name from object attributes or env vars."""
    for attr in ("model_name", "model", "DEFAULT_MODEL"):
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if isinstance(val, str) and val:
                return val
    if provider_key in ("vertexai", "genai", "gemini"):
        return os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    if provider_key == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return "unknown"


def _wrap_signals_only(
    fn: Callable[[str], UserEvent], provider_key: str
) -> ClassifierProvider:
    """Wrap legacy (text -> signals) into (signals, meta) with simple latency."""

    def _wrapped(event_text: str) -> Tuple[UserEvent, SignalMeta]:
        t0 = time.perf_counter()
        signals = fn(event_text)
        dur_ms = int((time.perf_counter() - t0) * 1000)
        meta: SignalMeta = {
            "provider": provider_key,
            "model": _model_from_obj_or_env(fn, provider_key),
            "latency_ms": dur_ms,
        }
        return signals, meta

    return FunctionProvider(_wrapped)


def _ensure_provider(obj: Any, provider_key: str) -> ClassifierProvider:
    """
    Normalize to (signals, meta).

    Supports:
      - object.classify(text) -> signals | (signals, meta)
      - callable(text) -> signals | (signals, meta)
    """

    def _attach_defaults(model_owner: Any, out: Any, dur_ms: int):
        if isinstance(out, dict):
            return out, {
                "provider": provider_key,
                "model": _model_from_obj_or_env(model_owner, provider_key),
                "latency_ms": dur_ms,
            }
        signals, meta = out
        meta = dict(meta or {})
        meta.setdefault("provider", provider_key)
        meta.setdefault("model", _model_from_obj_or_env(model_owner, provider_key))
        meta.setdefault("latency_ms", dur_ms)
        return signals, meta

    if hasattr(obj, "classify") and callable(getattr(obj, "classify")):

        def _call(event_text: str):
            t0 = time.perf_counter()
            out = obj.classify(event_text)
            dur_ms = int((time.perf_counter() - t0) * 1000)
            return _attach_defaults(obj, out, dur_ms)

        return FunctionProvider(_call)

    if callable(obj):

        def _call(event_text: str):
            t0 = time.perf_counter()
            out = obj(event_text)
            dur_ms = int((time.perf_counter() - t0) * 1000)
            return _attach_defaults(obj, out, dur_ms)

        return FunctionProvider(_call)

    raise TypeError("Unsupported provider type from get_llm_classifier().")


def main():
    parser = argparse.ArgumentParser(description="Panel Monitoring Agent")
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        default=os.getenv("LLM_PROVIDER", "openai"),
        help="Which LLM provider to use (default: openai)",
    )
    parser.add_argument(
        "--project",
        default=os.getenv("LANGSMITH_PROJECT", "panel-monitoring-agent"),
        help="LangSmith project name",
    )
    args = parser.parse_args()

    # Normalize alias: "gemini" -> "genai"
    provider_key = {"gemini": "genai"}.get(args.provider, args.provider)
    logger.info(f"Using provider: {args.provider} (resolved: {provider_key})")

    # Optional LangSmith project init
    try:
        Client().create_project(args.project, upsert=True)
        logger.info(f"Agent ready. LangSmith Project: {args.project}")
    except Exception:
        logger.warning("Agent ready. (LangSmith tracing disabled)")

    # Build provider
    try:
        raw_provider = get_llm_classifier(provider_key)
    except ValueError:
        logger.error(
            f"Unknown provider '{args.provider}'. Valid options: {', '.join(SUPPORTED_PROVIDERS)}"
        )
        sys.exit(2)
    except Exception as e:
        logger.error(f"Failed to initialize provider '{args.provider}': {e}")
        logger.error("Tip: check your API key/credentials env vars for this provider.")
        sys.exit(3)

    provider: ClassifierProvider
    if callable(raw_provider) and not hasattr(raw_provider, "classify"):
        provider = _wrap_signals_only(raw_provider, provider_key)
    else:
        provider = _ensure_provider(raw_provider, provider_key)

    # Build graph and start interactive loop
    try:
        app = build_graph()
    except Exception as e:
        logger.error(f"Failed to build graph: {e}")
        sys.exit(4)

    run_interactive(
        app,
        get_event_input=get_event_input,
        project_name=args.project,
        provider=provider,
    )


if __name__ == "__main__":
    main()
