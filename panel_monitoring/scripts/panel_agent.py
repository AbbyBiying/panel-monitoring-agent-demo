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
    """Best-effort model name extraction for metadata."""
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


def _wrap_signals_only(fn: Callable[[str], UserEvent], provider_key: str) -> ClassifierProvider:
    """Adapt legacy function (event -> signals) to (event -> (signals, meta)) with latency."""
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
    Ensure the provider yields (signals, meta).
    Supports:
      - callables returning UserEvent         -> wrap to (signals, meta) + latency
      - callables returning (signals, meta)     -> pass through (+ default meta if missing)
      - objects with .classify(...) either shape
    """
    # Object with .classify(...)
    if hasattr(obj, "classify") and callable(getattr(obj, "classify")):
        def _call(event_text: str):
            t0 = time.perf_counter()
            out = obj.classify(event_text)
            dur_ms = int((time.perf_counter() - t0) * 1000)

            if isinstance(out, dict):  # legacy signals-only
                return out, {
                    "provider": provider_key,
                    "model": _model_from_obj_or_env(obj, provider_key),
                    "latency_ms": dur_ms,
                }
            signals, meta = out
            meta = dict(meta or {})
            meta.setdefault("provider", provider_key)
            meta.setdefault("model", _model_from_obj_or_env(obj, provider_key))
            meta.setdefault("latency_ms", dur_ms)
            return signals, meta
        return FunctionProvider(_call)

    # Plain callable
    if callable(obj):
        def _call(event_text: str):
            t0 = time.perf_counter()
            out = obj(event_text)
            dur_ms = int((time.perf_counter() - t0) * 1000)

            if isinstance(out, dict):  # legacy signals-only
                return out, {
                    "provider": provider_key,
                    "model": _model_from_obj_or_env(obj, provider_key),
                    "latency_ms": dur_ms,
                }
            signals, meta = out
            meta = dict(meta or {})
            meta.setdefault("provider", provider_key)
            meta.setdefault("model", _model_from_obj_or_env(obj, provider_key))
            meta.setdefault("latency_ms", dur_ms)
            return signals, meta
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

    # Normalize alias used elsewhere: "gemini" -> "genai"
    provider_key = {"gemini": "genai"}.get(args.provider, args.provider)
    logger.info(f"Using provider: {args.provider} (resolved: {provider_key})")

    # LangSmith project (optional)
    try:
        Client().create_project(
            args.project, upsert=True
        )  # see if we can just get the current project created
        logger.info(f"Agent ready. LangSmith Project: {args.project}")
    except Exception:
        logger.warning("Agent ready. (LangSmith tracing disabled)")

    # Get a provider (function or object) from your existing factory
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

    # Ensure it matches the new (signals, meta) protocol
    if callable(raw_provider) and not hasattr(raw_provider, "classify"):
        # Might be legacy signals-only callable
        provider: ClassifierProvider = _wrap_signals_only(raw_provider, provider_key)
    else:
        provider = _ensure_provider(raw_provider, provider_key)

    # Build graph and run interactive loop
    try:
        app = build_graph(provider)
    except Exception as e:
        logger.error(f"Failed to build graph: {e}")
        sys.exit(4)

    run_interactive(app, get_event_input=get_event_input, project_name=args.project)


if __name__ == "__main__":
    main()
