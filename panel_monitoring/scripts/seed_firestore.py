# panel_monitoring/scripts/seed_firestore.py

from __future__ import annotations

from datetime import datetime, timezone
from google.cloud import firestore
from panel_monitoring.data.firestore_client import events_col, projects_col, alerts_col


def main():
    project_id = "panel-app-dev"

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


if __name__ == "__main__":
    main()
