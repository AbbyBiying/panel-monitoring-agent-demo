# panel_monitoring/app/runtime.py
from __future__ import annotations

import logging
from datetime import datetime, UTC
from typing import Any, Callable, Dict

from langsmith.run_helpers import trace
from panel_monitoring.app.utils import parse_event_input  # same package

logger = logging.getLogger(__name__)


def _now_utc_label() -> str:
    """HH:MM:SS label in UTC (avoid deprecated utcnow)."""
    return datetime.now(UTC).strftime("%H:%M:%S")


def run_interactive(app, get_event_input: Callable[[], str], project_name: str) -> None:
    """Simple REPL for manual runs."""
    logger.info("\n" + "=" * 50)
    logger.info("PANEL MONITORING AGENT READY")
    logger.info("Paste JSON or text. Type ':paste' for multi-line (end with 'END').")
    logger.info("Press Ctrl+C to exit.")
    logger.info("=" * 50)

    while True:
        try:
            raw = get_event_input()
            if not raw:
                logger.info(
                    "No input—try again. Use ':paste' for multi-line; end with 'END'."
                )
                continue

            try:
                event = parse_event_input(raw)  # dict payload from text/JSON
            except Exception as e:
                logger.error(f"Input parse/validation error: {e}")
                continue

            with trace(
                name=f"Manual Run - {_now_utc_label()} UTC", project_name=project_name
            ):
                # Invoke the compiled LangGraph app; state is a plain dict patch
                state: Dict[str, Any] = app.invoke({"event_data": event})

            # --- Short summary ---
            cls = state.get("classification", "error")
            act = state.get("action") or ("no_action" if cls == "normal" else "N/A")
            conf = state.get("confidence")
            conf_txt = f"{float(conf):.2f}" if isinstance(conf, (int, float)) else "N/A"
            expl = state.get("explanation_report") or ""

            logger.info("\n--- SUMMARY ---")
            logger.info(
                f"Classification: {cls} | Confidence: {conf_txt} | Action: {act}"
            )
            if expl:
                logger.info(f"Explanation: {expl}")

            # --- Full log entry (JSON pretty string) ---
            log_entry = state.get("log_entry")
            if log_entry:
                logger.info("\n--- FULL LOG ---")
                logger.info(log_entry)

        except KeyboardInterrupt:
            logger.info("\nExiting agent.")
            break
