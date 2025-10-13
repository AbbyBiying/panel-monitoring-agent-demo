# panel_monitoring/scripts/panel_agent.py
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from functools import partial
from typing import Any, Callable, Tuple, Union

from panel_monitoring.app.graph import build_graph
from panel_monitoring.app.runtime import run_interactive
from panel_monitoring.app.utils import get_event_input
from panel_monitoring.app.clients.llms import get_llm_classifier
from panel_monitoring.app.clients.llms.provider_base import (
    ClassifierProvider,
    FunctionProvider,
    Signal,
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


def _normalize_call(
    fn: Callable[[str], Union[Signal, Tuple[Signal, SignalMeta]]],
    owner: Any,
    provider_key: str,
    text: str,
) -> Tuple[Signal, SignalMeta]:
    t0 = time.perf_counter()
    out = fn(text)
    dur_ms = int((time.perf_counter() - t0) * 1000)
    if isinstance(out, dict):
        signals, meta = out, {}
    else:
        signals, meta = out  # type: ignore[assignment]
    meta = dict(meta or {})
    meta.setdefault("provider", provider_key)
    meta.setdefault("model", _model_from_obj_or_env(owner, provider_key))
    meta.setdefault("latency_ms", dur_ms)
    return signals, meta


def _wrap_provider(raw: Any, provider_key: str) -> ClassifierProvider:
    call = raw.classify if hasattr(raw, "classify") and callable(raw.classify) else raw
    if not callable(call):
        raise TypeError("Unsupported provider type from get_llm_classifier().")
    bound = partial(_normalize_call, call, raw, provider_key)
    return FunctionProvider(bound)


def main():
    p = argparse.ArgumentParser(description="Panel Monitoring Agent")
    p.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        default=os.getenv("PANEL_DEFAULT_PROVIDER", "openai"),
        help="LLM provider to use (default from PANEL_DEFAULT_PROVIDER).",
    )
    p.add_argument(
        "--project",
        default=os.getenv("LANGSMITH_PROJECT", "panel-monitoring-agent"),
        help="LangSmith project name.",
    )
    args = p.parse_args()

    provider_key = {"gemini": "genai"}.get(args.provider, args.provider)
    logger.info(f"Using provider: {args.provider} (resolved: {provider_key})")

    try:
        raw = get_llm_classifier(provider_key)
        provider = _wrap_provider(raw, provider_key)
    except ValueError:
        logger.error(
            f"Unknown provider '{args.provider}'. "
            f"Valid options: {', '.join(SUPPORTED_PROVIDERS)}"
        )
        sys.exit(2)
    except Exception as e:
        logger.error(f"Failed to initialize provider '{args.provider}': {e}")
        sys.exit(3)

    try:
        app = build_graph()
    except Exception as e:
        logger.error(f"Failed to build graph: {e}")
        sys.exit(4)

    logger.info("Agent ready.")
    run_interactive(
        app,
        get_event_input=get_event_input,
        project_name=args.project,
        provider=provider,
    )


if __name__ == "__main__":
    main()
