# panel_monitoring/scripts/smoke_datastore.py
from __future__ import annotations

import os
import logging
import asyncio # Added for async support
from datetime import datetime, timezone
from dotenv import load_dotenv
from google.cloud import firestore # Used for SERVER_TIMESTAMP

from panel_monitoring.data.firestore_client import get_db, events_col, runs_col

logger = logging.getLogger(__name__)

def _project() -> str | None:
    return (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
        or os.getenv("GCP_PROJECT_ID")
    )

async def run_smoke_test(): # Logic moved into async function
    load_dotenv()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    project_id = _project()
    if not project_id:
        raise SystemExit(
            "No project resolved. Set GOOGLE_CLOUD_PROJECT (or GCP_PROJECT*)."
        )

    source = os.getenv("PM_SOURCE_ID", "S1")
    logger.info("Smoke datastore start: project=%s source=%s", project_id, source)

    # Ensure client init works - must be AWAITED
    db = await get_db()
    logger.info("Firestore client initialized (Async)")

    # ---------------- 1) Create/Upsert an event (TOP-LEVEL 'events') ----------------
    masked_payload = {"content": "masked: user *** emailed ***"}
    meta = {"ip": "1.2.3.4", "ua": "smoke"}

    # Get collection reference - must be AWAITED
    col_ref = await events_col()
    evt_ref = col_ref.document()  # auto-id
    event_id = evt_ref.id
    
    # Writing to DB - must be AWAITED
    await evt_ref.set(
        {
            "project_id": project_id,
            "source_id": source,
            "type": "signup",
            "source": "web",
            "received_at": firestore.SERVER_TIMESTAMP,
            "event_at": datetime.now(timezone.utc),
            "status": "pending",
            "payload": masked_payload,
            "meta": meta,
        }
    )
    logger.info("Event upserted: id=%s", event_id)

    # ---------------- 2) Add a run (TOP-LEVEL 'runs') ----------------
    stats = {
        "status": "success",
        "prompt_hash": "demo",
        "input_count": 1,
        "output_count": 1,
        "input_cost_usd": 0.001,
        "output_cost_usd": 0.001,
        "total_cost_usd": 0.002,
        "request": {},
        "response": {},
        "meta": {},
    }

    # Get collection reference - must be AWAITED
    run_col_ref = await runs_col()
    run_ref = run_col_ref.document()  # auto-id
    attempt_id = run_ref.id
    
    # Writing to DB - must be AWAITED
    await run_ref.set(
        {
            "project_id": project_id,
            "event_id": event_id,
            "provider": "openai",
            "model_name": "gpt-4.1-mini",
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
            **stats,
        }
    )
    logger.info("Run added: attempt=%s", attempt_id)

    # ---------------- 3) Finalize event (update the event doc) ----------------
    finalize_fields = {
        "status": "ok",
        "decision": "normal",
        "confidence": 0.9,
        "duration_ms": 100,
        "last_run_provider": "openai",
        "last_run_model": "gpt-4.1-mini",
        "last_run_total_cost_usd": stats["total_cost_usd"],
        "updated_at": firestore.SERVER_TIMESTAMP,
        "last_run_id": attempt_id,
    }
    # Finalize - must be AWAITED
    await evt_ref.set(finalize_fields, merge=True)
    logger.info("Event finalized: id=%s", event_id)

def main():
    # Use asyncio.run to bridge sync main to async logic
    asyncio.run(run_smoke_test())

if __name__ == "__main__":
    main()