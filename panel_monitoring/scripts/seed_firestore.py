# panel_monitoring/scripts/seed_firestore.py

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv
from google.cloud import firestore
from panel_monitoring.data.firestore_client import events_col, projects_col, alerts_col, prompt_specs_col

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

    # 4. Seed PromptSpec (signup_classification — live)
    ps_col = await prompt_specs_col()
    ps_ref = ps_col.document("signup_classification_v1")
    await ps_ref.set(
        {
            "model_host": "vertexai",
            "model_name": "gemini-2.5-flash",
            "system_prompt": (
                "You are a Senior Fraud & Abuse Detection Expert for a consumer survey panel.\n"
                f"Assume the current date is {datetime.now(timezone.utc).strftime('%B %d, %Y')}.\n"
                "\n"
                "### HARD OVERRIDES\n"
                "If 'rule_based_flags' contains 'Failed manual validation' or similar, you MUST "
                "classify as 'suspicious_signup'. A human decision always overrides AI scoring.\n"
                "\n"
                "### ANALYSIS\n"
                "Use the Relevant Business Context provided to understand flag definitions, thresholds, "
                "and domain terminology. Evaluate: Identity Integrity, Geographical Alignment, "
                "Intent & Logical Consistency, and Technical Signals.\n"
                "\n"
                "### VERDICT\n"
                "- If evidence is insufficient or contradictory, lean towards 'normal_signup'.\n"
                "- ALWAYS return only the JSON schema provided.\n"
            ),
            "prompt": (
                "Analyze the following panelist event data for signals of fraud or abuse.\n"
                "\n"
                '1. Return the classification in the provided JSON schema. Ensure exactly one of "suspicious_signup" or "normal_signup" is true.\n'
                '2. "suspicious_signup" must be TRUE if the event shows characteristics of bots, identity theft, or policy abuse '
                "(e.g., using a disposable email explicitly listed, rapid-fire activity, or known fraudulent keywords present).\n"
                '3. "normal_signup" must be TRUE ONLY if the event appears to be from a legitimate, unique user.\n'
                "\n"
                "Event Data:\n"
                "{event}\n"
            ),
            "config": {"temperature": 0},
            "version": "1",
            "labels": ["fraud", "signup"],
            "deployment_status": "live",
            "deployment_role": "signup_classification",
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )
    logger.info("[ok] prompt_spec %s seeded", ps_ref.id)


def main():
    """Entry point using asyncio bridge."""
    try:
        asyncio.run(run_seed())
    except Exception as e:
        logger.error("Seeding failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
