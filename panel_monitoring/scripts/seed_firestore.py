# panel_monitoring/scripts/seed_firestore.py

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv
from google.cloud import firestore
from panel_monitoring.data.firestore_client import events_col, projects_col, alerts_col

# Setup basic logging to see progress
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

async def run_seed():
    """Asynchronous seeding logic for Firestore."""
    project_id = "panel-app-dev"

    # 1. Seed Project Metadata
    # We await the collection helper to get the reference
    p_col = await projects_col()
    proj_ref = p_col.document(project_id)
    
    # We await the .set() operation
    await proj_ref.set(
        {
            "name": "Panel Monitoring Agent",
            "status": "active",
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )
    logger.info("[ok] project %s seeded", project_id)

    # 2. Seed an Event
    e_col = await events_col()
    evt_ref = e_col.document()
    await evt_ref.set(
        {
            "project_id": project_id,
            "type": "signup",
            "source": "web",
            "received_at": firestore.SERVER_TIMESTAMP,
            "event_at": datetime.now(timezone.utc),
            "user_hash": "uh_demo",
            "ip_hash": "ih_demo",
            "payload": {"email_masked": "t***@e***.com", "ua_family": "Chrome"},
            "status": "pending",
        }
    )
    logger.info("[ok] event %s seeded for project %s", evt_ref.id, project_id)

    # 3. Seed an Alert
    a_col = await alerts_col()
    alert_ref = a_col.document()
    await alert_ref.set(
        {
            "project_id": project_id,
            "level": "info",
            "message": "Seed alert for testing",
            "created_at": firestore.SERVER_TIMESTAMP,
        }
    )
    logger.info("[ok] alert %s seeded for project %s", alert_ref.id, project_id)

def main():
    """Entry point using asyncio bridge."""
    try:
        asyncio.run(run_seed())
    except Exception as e:
        logger.error("Seeding failed: %s", e)
        raise SystemExit(1)

if __name__ == "__main__":
    main()