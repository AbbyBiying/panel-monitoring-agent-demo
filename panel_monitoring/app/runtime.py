# panel_monitoring/app/runtime.py
from __future__ import annotations

import logging
from datetime import datetime

from langsmith.run_helpers import trace
from panel_monitoring.app.schemas import GraphState
from panel_monitoring.app.utils import parse_event_input  # same package

logger = logging.getLogger(__name__)


def run_interactive(app, get_event_input, project_name: str, provider):
    """Shared REPL loop for manual runs."""
    logger.info("\n" + "=" * 50)
    logger.info("PANEL MONITORING AGENT READY")
    logger.info("Paste JSON or text. Type ':paste' for multi-line (end with 'END').")
    logger.info("=" * 50)

    while True:
        try:
            raw = get_event_input()
            if not raw:
                logger.info(
                    "No input—try again. Type ':paste' for multi-line; end with 'END'."
                )
                continue

            try:
                event = parse_event_input(raw)  # may be str OR dict
            except Exception as e:
                logger.error(f"Input parse/validation error: {e}")
                continue

            payload = (
                {"event_text": event}
                if isinstance(event, str)
                else {"event_data": event}
            )

            with trace(
                name=f"Manual Run - {datetime.now().strftime('%H:%M:%S')}",
                project_name=project_name,
            ):
                result = app.invoke(payload, context={"provider": provider})

            state = result if isinstance(result, GraphState) else GraphState(**result)

            logger.info("\n--- EXPLANATION ---")
            logger.info(state.explanation_report or "No explanation.")

            logger.info("\n--- FULL LOG ---")
            logger.info(state.log_entry or "")

        except KeyboardInterrupt:
            logger.info("\nExiting agent.")
            break
