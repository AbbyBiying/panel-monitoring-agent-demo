# panel_monitoring/scripts/seed_firestore.py

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from google.cloud import firestore
from panel_monitoring.data.firestore_client import events_col, projects_col, alerts_col


logger = logging.getLogger(__name__)

<<<<<<< Updated upstream

def _project() -> str | None:
    return (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
        or os.getenv("GCP_PROJECT_ID")
    )


def main():
    project = _project()

    if not project:
        raise SystemExit(
            "No project resolved. Set GOOGLE_CLOUD_PROJECT (or GCP_PROJECT*)."
        )
    # Create/merge a project doc
    project_ref = project_doc(project)
    project_ref.set({
        "name": "Panel Monitoring Agent",
        "status": "active",
        "created_at": firestore.SERVER_TIMESTAMP,
        "updated_at": firestore.SERVER_TIMESTAMP,
    }, merge=True)
    print(f"[ok] project {project} seeded")

    # Create one event doc
    evt_ref = events_col(project).document()
    evt_ref.set({
        "type": "signup",
        "source": "web",
        "received_at": firestore.SERVER_TIMESTAMP,
        "event_at": datetime.now(timezone.utc),
        "user_hash": "uh_demo",
        "ip_hash": "ih_demo",
        "payload": {"email_masked": "t***@e***.com", "ua_family": "Chrome"},
        "status": "pending",
    })
    print(f"[ok] event {evt_ref.id} seeded under project {project}")
=======
    # Create/merge a project doc in a flat collection
    proj_ref = projects_col().document(project_id)
    proj_ref.set(
        {
            "name": "Panel Monitoring Agent",
            "status": "active",
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )
    print(f"[ok] project {project_id} seeded")

    # Create one event doc (top-level events collection, project_id is just a field)
    evt_ref = events_col().document()
    evt_ref.set(
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
    print(f"[ok] event {evt_ref.id} seeded for project {project_id}")

    # Create one alert doc
    alert_ref = alerts_col().document()
    alert_ref.set(
        {
            "project_id": project_id,
            "level": "info",
            "message": "Seed alert for testing",
            "created_at": firestore.SERVER_TIMESTAMP,
        }
    )
    print(f"[ok] alert {alert_ref.id} seeded for project {project_id}")

>>>>>>> Stashed changes

if __name__ == "__main__":
    main()
