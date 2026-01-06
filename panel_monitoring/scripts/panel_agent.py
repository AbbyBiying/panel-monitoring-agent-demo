# panel_monitoring/scripts/panel_agent.py
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from panel_monitoring.app.graph import build_graph
from panel_monitoring.app.runtime import run_interactive
from panel_monitoring.app.utils import get_event_input
from panel_monitoring.app.clients.llms import init_llm_client

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("openai", "genai", "gemini", "vertexai")

def main():
    p = argparse.ArgumentParser(description="Panel Monitoring Agent")
    p.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        default=os.getenv("PANEL_DEFAULT_PROVIDER", "vertexai"),
        help="LLM provider to use (default from PANEL_DEFAULT_PROVIDER).",
    )
    p.add_argument(
        "--project",
        default=os.getenv("LANGSMITH_PROJECT", "panel-monitoring-agent"),
        help="LangSmith project name.",
    )
    args = p.parse_args()

    # Normalize provider key
    provider_key = {"gemini": "genai"}.get(args.provider, args.provider)
    logger.info(f"Using provider: {args.provider} (resolved: {provider_key})")

    try:
        # 1. Initialize the Async Singleton for the chosen provider
        init_llm_client(provider_key)
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
        # 2. Build the LangGraph
        app = build_graph()
    except Exception as e:
        logger.error(f"Failed to build graph: {e}")
        sys.exit(4)

    logger.info("Agent ready (Async Mode).")

    # 3. Run the async event loop
    asyncio.run(run_interactive(
        app,
        get_event_input=get_event_input,
        project_name=args.project,
        provider=provider_key,
    ))

if __name__ == "__main__":
    main()