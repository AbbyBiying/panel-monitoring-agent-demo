# panel_monitoring/scripts/seed_firestore.py

from __future__ import annotations

from datetime import datetime, timezone
from google.cloud import firestore
from panel_monitoring.data.firestore_client import project_doc, events_col

def main():
    project_id = "panel-app-dev"

    # Create/merge a project doc
    project_ref = project_doc(project_id)
    project_ref.set({
        "name": "Panel Monitoring Agent",
        "status": "active",
        "created_at": firestore.SERVER_TIMESTAMP,
        "updated_at": firestore.SERVER_TIMESTAMP,
    }, merge=True)
    print(f"[ok] project {project_id} seeded")

    # Create one event doc
    evt_ref = events_col(project_id).document()
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
    print(f"[ok] event {evt_ref.id} seeded under project {project_id}")

if __name__ == "__main__":
    main()
