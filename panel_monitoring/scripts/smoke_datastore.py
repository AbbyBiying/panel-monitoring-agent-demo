# panel_monitoring/scripts/smoke_datastore.py
from __future__ import annotations
import os, logging, time
from dotenv import load_dotenv
from panel_monitoring.data.firestore_client import get_db, metrics_daily_doc
from panel_monitoring.data.ingest import add_run, finalize_event, upsert_event

logger = logging.getLogger(__name__)

def _project() -> str | None:
    return (os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("GCLOUD_PROJECT")
            or os.getenv("GCP_PROJECT")
            or os.getenv("GCP_PROJECT_ID"))

def main():
    load_dotenv()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    project = _project()
    if not project:
        raise SystemExit("No project resolved. Set GOOGLE_CLOUD_PROJECT (or GCP_PROJECT*).")

    source = os.getenv("PM_SOURCE_ID", "S1")
    logger.info("Smoke datastore start: project=%s source=%s", project, source)

    # Ensure client init works
    get_db()
    logger.info("Firestore client initialized")

    # 1) Upsert a tiny masked event
    masked_payload = {"content": "masked: user *** emailed ***"}
    meta = {"ip": "1.2.3.4", "ua": "smoke"}
    event_ref, event_id = upsert_event(project, source, masked_payload, meta)
    logger.info("Event upserted: id=%s", event_id)

    # 2) Add a run
    stats = {
        "status": "success", "prompt_hash": "demo",
        "input_count": 1, "output_count": 1,
        "input_cost_usd": 0.001, "output_cost_usd": 0.001, "total_cost_usd": 0.002,
        "request": {}, "response": {}, "meta": {}
    }
    run_ref, attempt = add_run(event_ref, "openai", "gpt-4.1-mini", stats)
    logger.info("Run added: attempt=%s", attempt)

    # 3) Finalize event
    finalize_event(project, event_ref, run_ref, "ok", 0.9, 100, "openai", stats["total_cost_usd"])
    logger.info("Event finalized")

    # 4) Read daily metrics
    day = time.strftime("%Y-%m-%d", time.gmtime())
    doc = metrics_daily_doc(project, day).get().to_dict()
    logger.info("Daily metrics %s: %s", day, doc or "{}")

    logger.info("Smoke datastore: SUCCESS")

if __name__ == "__main__":
    main()
