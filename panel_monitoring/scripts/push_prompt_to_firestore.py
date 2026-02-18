# panel_monitoring/scripts/push_prompt_to_firestore.py
"""
Push the local prompts.py content to Firestore as a new versioned PromptSpec.

PromptSpec documents are IMMUTABLE after creation. Each push creates a new
document so that every Run can trace back to the exact prompt that was used.
Editing an existing PromptSpec would silently invalidate past audit logs.

Workflow:
    1. Run this script  →  creates a new doc with deployment_status = "pre_live"
    2. Review the new prompt in Firestore
    3. Promote manually:  set deployment_status → "canary" or "live"
    4. Old versions stay in Firestore forever — never delete them

Usage:
    uv run python -m panel_monitoring.scripts.push_prompt_to_firestore
"""
from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from panel_monitoring.app.prompts import PROMPT_CLASSIFY_SYSTEM, PROMPT_CLASSIFY_USER
from panel_monitoring.data.firestore_client import prompt_specs_col

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

DEPLOYMENT_ROLE = "signup_classification"


async def push():
    col = await prompt_specs_col()

    # Find the highest existing version for this role.
    # We never edit existing docs — we always create a new one.
    query = col.where(filter=FieldFilter("deployment_role", "==", DEPLOYMENT_ROLE))
    docs = [doc async for doc in query.stream()]

    max_version = 0
    for doc in docs:
        data = doc.to_dict() or {}
        try:
            v = int(data.get("version", "0"))
            max_version = max(max_version, v)
        except (ValueError, TypeError):
            pass

    new_version = max_version + 1
    new_doc_id = f"{DEPLOYMENT_ROLE}_v{new_version}"

    # Create a brand-new document. Never use merge=True on an existing doc.
    ref = col.document(new_doc_id)
    await ref.set(
        {
            "system_prompt": PROMPT_CLASSIFY_SYSTEM,
            "user_prompt": PROMPT_CLASSIFY_USER,
            "version": str(new_version),
            "deployment_role": DEPLOYMENT_ROLE,
            "deployment_status": "pre_live",
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }
    )

    logger.info("[ok] Created PromptSpec '%s' (version=%s, status=pre_live)", new_doc_id, new_version)
    logger.info("  Next steps:")
    logger.info("    canary → set deployment_status = 'canary' in Firestore")
    logger.info("    live   → set deployment_status = 'live'   in Firestore")
    logger.info("  Do NOT edit or delete older versions — past runs reference them.")


def main():
    try:
        asyncio.run(push())
    except Exception as e:
        logger.error("Push failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
