# panel_monitoring/data/firestore_client.py
from __future__ import annotations

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from google.cloud import firestore
from google.auth.exceptions import DefaultCredentialsError

from panel_monitoring.app.utils import load_credentials, log_info, make_credentials_from_env

logger = logging.getLogger(__name__)

load_dotenv()

_DB: Optional[firestore.Client] = None


def get_db() -> firestore.Client:
    """
    Return a singleton Firestore client with support for emulator and non-default DB.

    Environment variables:
        GCP_PROJECT or GOOGLE_CLOUD_PROJECT - Firestore project ID
        FIRESTORE_DATABASE_ID - Optional, defaults to "(default)"
        FIRESTORE_EMULATOR_HOST - Optional, if using emulator
        GOOGLE_APPLICATION_CREDENTIALS - Path to credentials file
    """
    global _DB
    if _DB is not None:
        return _DB

    project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    database = os.getenv("FIRESTORE_DATABASE_ID", "panel-monitoring-agent-dev")

    if os.getenv("ENVIRONMENT") == "local":
        logger.info("Running in LOCAL environment, loading credentials from file.")
        log_info("Running in local environment, loading credentials from file.")
        creds = load_credentials()
    else:
        log_info("Running in NOT LOCAL environment, loading credentials from Path.")
        creds = make_credentials_from_env()

        logger.info(
            "Running in NOT LOCAL environment, loading credentials from Path."
        )
    logger.debug("creds type: %s", type(creds))

    try:
        _DB = firestore.Client(project=project, database=database, credentials=creds)
        logger.info(
            "Initialized Firestore client for project '%s' (DB: %s)", project, database
        )
    except DefaultCredentialsError as e:
        raise RuntimeError(
            "Firestore credentials not found. Please set GOOGLE_APPLICATION_CREDENTIALS "
            "and GCP_PROJECT/GOOGLE_CLOUD_PROJECT, or run `gcloud auth application-default login`, "
            "or set FIRESTORE_EMULATOR_HOST."
        ) from e

    return _DB


def events_col() -> firestore.CollectionReference:
    """Return reference to top-level 'events' collection. Each doc includes `project_id`."""
    return get_db().collection("events")


def runs_col() -> firestore.CollectionReference:
    """Return reference to top-level 'runs' collection. Each doc includes `project_id` and `event_id`."""
    return get_db().collection("runs")


def alerts_col() -> firestore.CollectionReference:
    """Return reference to top-level 'alerts' collection. Each doc includes `project_id`."""
    return get_db().collection("alerts")


def projects_col() -> firestore.CollectionReference:
    """Return reference to flat 'projects' collection for metadata."""
    return get_db().collection("projects")