# panel_monitoring/scripts/peek_firestore.py

from __future__ import annotations
import asyncio
import logging
from dotenv import load_dotenv
from google.cloud import firestore
from panel_monitoring.data.firestore_client import events_col

# Set up logging to match the rest of your scripts
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def peek_latest(project_id: str = "panel-app-dev", window: int = 50):
    """
    Asynchronously fetches the latest events for a specific project.
    """
    # 1. Ensure env is loaded for standalone execution
    load_dotenv()

    # 2. Get the async collection reference
    col = await events_col()

    # 3. Build the query
    # Note: .stream() in AsyncClient returns an AsyncIterator
    query = col.order_by("received_at", direction=firestore.Query.DESCENDING).limit(
        window
    )

    logger.info("Peeking at last %d events for project: %s", window, project_id)

    # 4. Use 'async for' to iterate over the stream
    found = False
    async for d in query.stream():
        data = d.to_dict() or {}
        if data.get("project_id") == project_id:
            # Format the output for readability
            print(f"ID: {d.id}")
            print(f"Data: {data}\n")
            found = True
            return d.id

    if not found:
        logger.info(
            "No events found for project %s in the last %d records.", project_id, window
        )


def main():
    """Entry point using the asyncio bridge."""
    try:
        asyncio.run(peek_latest())
    except Exception as e:
        logger.error("Peek failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
