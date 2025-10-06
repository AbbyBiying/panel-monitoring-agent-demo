# panel_monitoring/data/firestore_client.py
from __future__ import annotations
import os
from typing import Optional
from dotenv import load_dotenv

from google.cloud import firestore
from google.auth.exceptions import DefaultCredentialsError
import logging

logger = logging.getLogger(__name__)


def _project() -> str | None:
    return (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
        or os.getenv("GCP_PROJECT_ID")
    )

load_dotenv()

_DB: Optional[firestore.Client] = None


def get_db() -> firestore.Client:
    """Singleton Firestore client with support for non-default DB and emulator."""
    global _DB
    if _DB is not None:
        return _DB

<<<<<<< Updated upstream
    emulator = os.getenv("FIRESTORE_EMULATOR_HOST")
    project = _project()
=======
    project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
>>>>>>> Stashed changes
    database = os.getenv("FIRESTORE_DATABASE_ID") or "(default)"

    try:
        from panel_monitoring.app.utils import load_credentials

        creds = load_credentials()
    except Exception:
        creds = None

    try:
        _DB = firestore.Client(
            project=project,
            database=database,
            credentials=creds,
        )
    except DefaultCredentialsError as e:
        raise RuntimeError(
            "Firestore creds not found. Set GOOGLE_APPLICATION_CREDENTIALS and GCP_PROJECT/GOOGLE_CLOUD_PROJECT, "
            "or run `gcloud auth application-default login`, or set FIRESTORE_EMULATOR_HOST."
        ) from e

    return _DB


def events_col():
    """Top-level 'events' collection. Each doc should include `project_id`."""
    return get_db().collection("events")


def runs_col():
    """Top-level 'runs' collection. Each doc should include `project_id` and `event_id`."""
    return get_db().collection("runs")


def alerts_col():
    """Top-level 'alerts' collection. Each doc should include `project_id`."""
    return get_db().collection("alerts")


# (Optional) keep projects metadata as a flat collection, but no subcollections
def projects_col():
    return get_db().collection("projects")
