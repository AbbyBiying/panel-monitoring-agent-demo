# panel-monitoring/scripts/peek_firestore.py

from panel_monitoring.data.firestore_client import events_col


def peek_latest(project_id: str = "panel-app-dev") -> str | None:
    """Print and return the most recent event ID from Firestore."""
    q = events_col(project_id).order_by("received_at", direction="DESCENDING").limit(1)
    docs = list(q.stream())
    if not docs:
        print("No events found for project:", project_id)
        return None

    latest = docs[0]
    print("LATEST EVENT:", latest.id, latest.to_dict())
    return latest.id

if __name__ == "__main__":
    peek_latest()