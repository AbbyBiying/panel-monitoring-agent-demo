# panel_monitoring/app/runtime.py
from __future__ import annotations

import logging
from datetime import datetime
from uuid import uuid4

from langsmith.run_helpers import trace
from langgraph.types import Command  # for resume after interrupt

from panel_monitoring.app.schemas import GraphState
from panel_monitoring.app.utils import parse_event_input  # same package

logger = logging.getLogger(__name__)


def run_interactive(app, get_event_input, project_name: str, provider: str):
    """
    Shared REPL loop for manual runs.
    - Creates a unique event_id per run and uses it as thread_id (for checkpoint/interrupt).
    - Handles LangGraph interrupts by asking for human input and resuming.
    - Shows explanation and log for each completed run.
    """
    logger.info("\n" + "=" * 50)
    logger.info("PANEL MONITORING AGENT READY")
    logger.info("Paste JSON or text. Type ':paste' for multi-line (end with 'END').")
    logger.info("Type Ctrl-C to exit.")
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

            # Build initial payload
            event_id = uuid4().hex
            payload = (
                {"event_id": event_id, "event_text": event}
                if isinstance(event, str)
                else {"event_id": event_id, "event_data": event}
            )

            # Stable thread_id == event_id so we can resume the same run after interrupt
            config = {"configurable": {"thread_id": event_id}}

            with trace(
                name=f"Manual Run - {datetime.now().strftime('%H:%M:%S')}",
                project_name=project_name,
            ):
                ctx = {"provider": provider}

                # First invoke
                result = app.invoke(payload, context=ctx, config=config)

                # Drain all interrupts (if any)
                while isinstance(result, dict) and "__interrupt__" in result:
                    last = result["__interrupt__"][-1]
                    msg = last.value

                    logger.info("\n--- HUMAN REVIEW REQUIRED ---")
                    logger.info("Interrupt payload:")
                    logger.info(str(msg))

                    decision = (
                        input("\nApprove deletion? (approve/reject/escalate): ")
                        .strip()
                        .lower()
                    )
                    if decision not in {"approve", "reject", "escalate"}:
                        logger.info("Unrecognized input; defaulting to 'reject'.")
                        decision = "reject"

                    # Resume with SAME config and SAME context
                    result = app.invoke(
                        Command(resume=decision), context=ctx, config=config
                    )

            state = result if isinstance(result, GraphState) else GraphState(**result)

            logger.info("\n--- EXPLANATION ---")
            logger.info(state.explanation_report or "No explanation.")

            logger.info("\n--- FULL LOG ---")
            logger.info(state.log_entry or "")

        except KeyboardInterrupt:
            logger.info("\nExiting agent.")
            break
        except Exception as e:
            logger.exception(f"Runtime error: {e}")
