"""
panel_monitoring.data

Thin facade over Firestore datastore helpers so the rest of the app can do:
    from panel_monitoring.data import upsert_event, add_run, finalize_event

Environment:
- GOOGLE_APPLICATION_CREDENTIALS: path to a GCP service account JSON
- GCP_PROJECT or GOOGLE_CLOUD_PROJECT: host GCP project id (e.g., "panel-monitoring-agent")
- FIRESTORE_DATABASE_ID: Firestore database id (e.g., "panel-monitoring-agent-dev") or "(default)"
- FIRESTORE_EMULATOR_HOST: host:port for local emulator (optional)

Exports:
- Firestore client helpers: get_db, project_doc, events_col, runs_col, alerts_col, metrics_daily_doc
- Ingest helpers: upsert_event, add_run, finalize_event
- Utility: db_info (returns {'project': ..., 'database': ...})
"""
from __future__ import annotations
import os

from .firestore_client import (
    get_db,
    project_doc,
    events_col,
    runs_col,
    alerts_col,
    metrics_daily_doc,
)
from .ingest import (
    upsert_event,
    add_run,
    finalize_event,
)


def db_info() -> dict:
    """
    Quick diagnostics for where writes will go.
    Uses env for database id (preferred) and the client's resolved project.
    """
    db = get_db()
    return {
        "project": db.project,
        "database": os.getenv("FIRESTORE_DATABASE_ID", "(default)"),
    }


__all__ = [
    "get_db",
    "project_doc",
    "events_col",
    "runs_col",
    "alerts_col",
    "metrics_daily_doc",
    "upsert_event",
    "add_run",
    "finalize_event",
    "db_info",
]
