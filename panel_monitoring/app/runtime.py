# panel_monitoring/app/runtime.py
import logging
from datetime import datetime

from langsmith.run_helpers import trace
from panel_monitoring.app.utils import parse_event_input  # same package

logger = logging.getLogger(__name__)


def run_interactive(app, get_event_input, project_name: str):
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
                event = parse_event_input(raw)
            except Exception as e:
                logger.error(f"Input parse/validation error: {e}")
                continue

            with trace(
                name=f"Manual Run - {datetime.now().strftime('%H:%M:%S')}",
                project_name=project_name,
            ):
                state = app.invoke({"event_data": event})

            logger.info("\n--- FULL LOG ---")
            logger.info(state.get("log_entry"))

        except KeyboardInterrupt:
            logger.info("\nExiting agent.")
            break
