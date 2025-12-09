# panel_monitoring/scripts/smoke_auth_check.py

from __future__ import annotations
import os
import logging
from dotenv import load_dotenv
from google.cloud import firestore
from panel_monitoring.app.utils import load_credentials

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

    creds = load_credentials()
    project = _project()
    if not project:
        raise RuntimeError(
            "No project resolved. Set GOOGLE_CLOUD_PROJECT (or GCP_PROJECT*)."
        )

    db_id = os.getenv("FIRESTORE_DATABASE_ID", "panel-monitoring-agent-dev"),
    client = firestore.Client(project=project, database=db_id, credentials=creds)
    logger.info("Starting Firestore smoke check: project=%s db=%s", project, db_id)

    try:
        for coll in client.collections():
            logger.info("Found collection: %s", coll.id)
            break
        else:
            logger.info("Reachable; no collections yet.")
        logger.info("Firestore smoke check: SUCCESS")
    except Exception:
        logger.exception("Firestore access failed")
        raise


if __name__ == "__main__":
    main()
