# panel_monitoring/scripts/panel_agent.py

import argparse
import logging
import os
import sys

from langsmith import Client
from panel_monitoring.app.graph import build_graph
from panel_monitoring.app.runtime import run_interactive
from panel_monitoring.app.utils import get_event_input
from panel_monitoring.app.clients.llms import get_llm_classifier


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


SUPPORTED_PROVIDERS = ("openai", "genai", "gemini", "vertexai")


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

    # Normalize alias
    provider_key = {"gemini": "genai"}.get(args.provider, args.provider)
    logger.info(f"Using provider: {args.provider}")

    try:
        Client().create_project(args.project, upsert=True)
        logger.info(f"Agent ready. LangSmith Project: {args.project}")
    except Exception:
        logger.warning("Agent ready. (LangSmith tracing disabled)")

    try:
        classify = get_llm_classifier(provider_key)
    except ValueError:
        logger.error(
            f"Unknown provider '{args.provider}'. Valid options: {', '.join(SUPPORTED_PROVIDERS)}"
        )
        sys.exit(2)
    except Exception as e:
        logger.error(f"Failed to initialize provider '{args.provider}': {e}")
        logger.error("Tip: check your API key/credentials env vars for this provider.")
        sys.exit(3)

    try:
        app = build_graph(classify)
    except Exception as e:
        logger.error(f"Failed to build graph: {e}")
        sys.exit(4)

    # Run interactive loop (reads event via get_event_input)
    run_interactive(app, get_event_input=get_event_input, project_name=args.project)


if __name__ == "__main__":
    main()
