# panel_monitoring/scripts/smoke_datastore.py
from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv
from google.cloud import firestore

from panel_monitoring.data.firestore_client import get_db, events_col, runs_col

logger = logging.getLogger(__name__)


def _project() -> str | None:
    return (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
        or os.getenv("GCP_PROJECT_ID")
    )


def main():
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

    # Ensure client init works
    get_db()
    logger.info("Firestore client initialized")

    # ---------------- 1) Create/Upsert an event (TOP-LEVEL 'events') ----------------
    masked_payload = {"content": "masked: user *** emailed ***"}
    meta = {"ip": "1.2.3.4", "ua": "smoke"}

    evt_ref = events_col().document()  # auto-id
    event_id = evt_ref.id
    evt_ref.set(
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

    run_ref = runs_col().document()  # auto-id
    attempt_id = run_ref.id
    run_ref.set(
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
    evt_ref.set(finalize_fields, merge=True)
    logger.info("Event finalized: id=%s", event_id)


if __name__ == "__main__":
    main()
