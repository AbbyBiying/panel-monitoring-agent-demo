# gcp/events-report/main.py
"""FastAPI app for the Firestore events reporting page. Run from repo root:
  uv run uvicorn main:app --reload --app-dir gcp/events-report
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Any

# Ensure repo root is on path so panel_monitoring is importable
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from dotenv import load_dotenv  # noqa: E402
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from google.cloud import firestore  # noqa: E402

load_dotenv(os.path.join(_root, ".env"))

# Always use local gcloud login (ADC); never use a credentials file
os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

# Import after path fix
from panel_monitoring.data.firestore_client import events_col  # noqa: E402

app = FastAPI(title="Events report", description="Firestore events by type")

# Serve static files from ./static
STATIC_DIR = os.path.join(_here, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _serialize_doc(doc_id: str, data: dict[str, Any]) -> dict[str, Any]:
    """Convert document to JSON-serializable dict; timestamps to ISO strings."""
    out = {"id": doc_id, **data}
    for key in ("updated_at", "received_at", "event_at", "created_at"):
        if key in out and out[key] is not None:
            val = out[key]
            if hasattr(val, "isoformat"):
                out[key] = val.isoformat()
            elif hasattr(val, "timestamp"):  # Firestore timestamp
                dt = datetime.fromtimestamp(val.timestamp(), tz=timezone.utc)
                out[key] = dt.isoformat()
    return out


@app.get("/")
async def index():
    """Serve the reporting page."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.isfile(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


@app.get("/api/events")
async def get_events(
    event_type: str = Query(..., min_length=1, alias="type", description="Filter by event type"),
):
    """Query events collection by type, ordered by updated_at descending."""
    type_val = event_type.strip()
    if not type_val:
        raise HTTPException(status_code=400, detail="type is required")
    try:
        col = await events_col()
        query = col.where("type", "==", type_val).order_by(
            "updated_at", direction=firestore.Query.DESCENDING
        )
        docs = []
        async for doc in query.stream():
            data = doc.to_dict() or {}
            docs.append(_serialize_doc(doc.id, data))
        return {"events": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
