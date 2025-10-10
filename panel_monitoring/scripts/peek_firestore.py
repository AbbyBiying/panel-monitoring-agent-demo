# panel_monitoring/scripts/peek_firestore.py

from __future__ import annotations
from google.cloud import firestore
from panel_monitoring.data.firestore_client import events_col

def peek_latest(project_id: str = "panel-app-dev", window: int = 50):
    # No composite index required: only order_by on received_at
    docs = (
        events_col()
        .order_by("received_at", direction=firestore.Query.DESCENDING)
        .limit(window)
        .stream()
    )
    for d in docs:
        data = d.to_dict() or {}
        if data.get("project_id") == project_id:
            print(d.id, data)
            return d.id

if __name__ == "__main__":
    peek_latest()
