# panel_monitoring/data/firestore_client.py
from __future__ import annotations

import asyncio
import os
import logging
from typing import Optional

from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1.async_collection import AsyncCollectionReference
from google.auth.exceptions import DefaultCredentialsError

from panel_monitoring.app.utils import load_credentials, log_info, make_credentials_from_env

logger = logging.getLogger(__name__)

_DB: Optional[AsyncClient] = None
_DB_LOCK = asyncio.Lock()


async def get_db() -> AsyncClient:
    """
    Return a singleton async Firestore client with support for emulator and non-default DB.

    Environment variables:
        GCP_PROJECT or GOOGLE_CLOUD_PROJECT - Firestore project ID
        FIRESTORE_DATABASE_ID - Optional, defaults to "(default)"
        FIRESTORE_EMULATOR_HOST - Optional, if using emulator
        GOOGLE_APPLICATION_CREDENTIALS - Path to credentials file
    """
    global _DB
    if _DB is not None:
        return _DB

    async with _DB_LOCK:
        # Double-check after acquiring lock
        if _DB is not None:
            return _DB

        project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        database = os.getenv("FIRESTORE_DATABASE_ID", "panel-monitoring-agent-dev")

        # Load credentials in a thread to avoid blocking
        if os.getenv("ENVIRONMENT") == "local":
            logger.info("Running in LOCAL environment, loading credentials from file.")
            log_info("Running in local environment, loading credentials from file.")
            creds = await asyncio.to_thread(load_credentials)
        else:
            log_info("Running in NOT LOCAL environment, loading credentials from Path.")
            creds = await asyncio.to_thread(make_credentials_from_env)
            logger.info(
                "Running in NOT LOCAL environment, loading credentials from Path."
            )
        logger.debug("creds type: %s", type(creds))

        try:
            _DB = AsyncClient(project=project, database=database, credentials=creds)
            logger.info(
                "Initialized async Firestore client for project '%s' (DB: %s)", project, database
            )
        except DefaultCredentialsError as e:
            raise RuntimeError(
                "Firestore credentials not found. Please set GOOGLE_APPLICATION_CREDENTIALS "
                "and GCP_PROJECT/GOOGLE_CLOUD_PROJECT, or run `gcloud auth application-default login`, "
                "or set FIRESTORE_EMULATOR_HOST."
            ) from e

        return _DB


async def events_col() -> AsyncCollectionReference:
    """Return reference to top-level 'events' collection. Each doc includes `project_id`."""
    db = await get_db()
    return db.collection("events")


async def runs_col() -> AsyncCollectionReference:
    """Return reference to top-level 'runs' collection. Each doc includes `project_id` and `event_id`."""
    db = await get_db()
    return db.collection("runs")


async def alerts_col() -> AsyncCollectionReference:
    """Return reference to top-level 'alerts' collection. Each doc includes `project_id`."""
    db = await get_db()
    return db.collection("alerts")


async def projects_col() -> AsyncCollectionReference:
    """Return reference to flat 'projects' collection for metadata."""
    db = await get_db()
    return db.collection("projects")