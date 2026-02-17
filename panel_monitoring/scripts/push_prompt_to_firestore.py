# panel_monitoring/scripts/push_prompt_to_firestore.py
"""
Push the local prompts.py content to the live PromptSpec in Firestore.

Usage:
    uv run python -m panel_monitoring.scripts.push_prompt_to_firestore
"""
from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv
from google.cloud import firestore

from panel_monitoring.app.prompts import PROMPT_CLASSIFY_SYSTEM, PROMPT_CLASSIFY_USER
from panel_monitoring.data.firestore_client import prompt_specs_col

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

DOC_ID = "signup_classification_v1"


async def push():
    col = await prompt_specs_col()
    ref = col.document(DOC_ID)

    await ref.set(
        {
            "system_prompt": PROMPT_CLASSIFY_SYSTEM,
            "prompt": PROMPT_CLASSIFY_USER,
            "updated_at": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )
    logger.info("[ok] Pushed local prompts to Firestore doc '%s'", DOC_ID)

    # Read back to confirm
    snap = await ref.get()
    data = snap.to_dict()
    logger.info("  system_prompt: %.120s...", data.get("system_prompt", ""))
    logger.info("  prompt: %.120s...", data.get("prompt", ""))


def main():
    try:
        asyncio.run(push())
    except Exception as e:
        logger.error("Push failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
