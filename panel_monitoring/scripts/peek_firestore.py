# panel-monitoring/scripts/peek_firestore.py

from panel_monitoring.data.firestore_client import events_col

def peek_latest(project_id="panel-app-dev"):
    q = events_col(project_id).order_by("received_at", direction="DESCENDING").limit(1)
    docs = list(q.stream())
    for d in docs:
        print(d.id, d.to_dict())
    return docs[0].id if docs else None

peek_latest()
