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

    emulator = os.getenv("FIRESTORE_EMULATOR_HOST")
    project = _project()
    database = os.getenv("FIRESTORE_DATABASE_ID") or "(default)"

    if emulator and not project:
        project = "demo-dev"

    # Only load creds when we actually build the client
    creds = None
    if not emulator:
        try:
            # defer import to avoid circulars / import-time failures
            from panel_monitoring.app.utils import load_credentials

            creds = load_credentials() or None
        except Exception:
            creds = None  # fall back to ADC if available

    try:
        _DB = firestore.Client(
            project=project,
            database=database,
            credentials=creds,  # None => ADC
        )
    except DefaultCredentialsError as e:
        raise RuntimeError(
            "Firestore creds not found. Set GOOGLE_APPLICATION_CREDENTIALS and GCP_PROJECT/GOOGLE_CLOUD_PROJECT, "
            "or run `gcloud auth application-default login`, or set FIRESTORE_EMULATOR_HOST."
        ) from e

    return _DB


def project_doc(project_id: str):
    return get_db().collection("projects").document(project_id)


def events_col(project_id: str):
    return project_doc(project_id).collection("events")


def runs_col(project_id: str, event_id: str):
    return events_col(project_id).document(event_id).collection("runs")


def alerts_col(project_id: str):
    return project_doc(project_id).collection("alerts")


def metrics_daily_doc(project_id: str, day: str):
    return (
        get_db()
        .collection("metrics")
        .document(f"projects_{project_id}")
        .collection("daily")
        .document(day)
    )
