# Cloud Run Functions (Gen2 on Cloud Run) – Pub/Sub → LangSmith
# Deploy:
# gcloud functions deploy pubsub-to-langsmith \
#   --region=us-central1 \
#   --runtime=python313 \
#   --source=. \
#   --entry-point=pubsub_to_langsmith \
#   --trigger-topic=user-event-signups

import json
import logging
import os
import base64
import typing as t
from dotenv import load_dotenv

import functions_framework
from langsmith import Client

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

def _get_ls_client() -> Client:
    api_key = (
        os.getenv("LANGSMITH_API_KEY_CLOUD_FUNCTIONS")
        or os.getenv("LANGSMITH_API_KEY")
    )
    if not api_key:
        # Fail fast so the error is obvious in logs; Pub/Sub will retry if enabled 
        logger.error(
            {
                "msg": "Missing LangSmith API key in environment.",
                "checked": ["LANGSMITH_API_KEY_CLOUD_FUNCTIONS", "LANGSMITH_API_KEY"],
            }
        )
        raise RuntimeError("Missing LANGSMITH API key")
    return Client(api_key=api_key, api_url=LANGSMITH_ENDPOINT)


# Configure LangSmith client from env (prefer Secret Manager for the API key)
_LS = _get_ls_client()
_PROJECT = os.getenv("LANGSMITH_PROJECT")


def _decode_pubsub_payload(event_data: dict) -> t.Tuple[t.Union[dict, str, None], dict]:
    """
    Extracts and decodes Pub/Sub payload + useful metadata from an Eventarc CloudEvent.

    Eventarc (Pub/Sub) CloudEvent data shape:
    {
      "message": {
        "data": "base64-encoded",
        "messageId": "123",
        "orderingKey": "key-1",
        "publishTime": "2025-01-01T00:00:00Z",
        "attributes": { ... }   # optional
      },
      "subscription": "projects/.../subscriptions/..."
    }
    Return (payload, meta) where payload is dict if JSON, str if not JSON, or None.
   
    """
    msg = (event_data or {}).get("message") or {}
    data_b64 = msg.get("data")
    payload = None
    if data_b64:
        try:
            raw = base64.b64decode(data_b64).decode("utf-8")
            try:
                payload = json.loads(raw) # Try JSON first; if not JSON, keep as text
            except json.JSONDecodeError:
                payload = raw # keep as text

        except Exception as e:
            logger.warning(f"Error decoding base64 data: {e}")
            raise

    meta = {
        "pubsub_message_id": msg.get("messageId"),
        "pubsub_ordering_key": msg.get("orderingKey"),
        "pubsub_publish_time": msg.get("publishTime"),
        "pubsub_attributes": msg.get("attributes") or {},
    }
    return payload, meta


@functions_framework.cloud_event
def pubsub_to_langsmith(event):
    """
    Cloud Run Function entry point for Pub/Sub.
    - Uses CloudEvents signature (required by Eventarc).
    - Raises on failure so Eventarc/Pub/Sub will retry per policy.
    """
    try:
        logger.info(event)
        payload, meta = _decode_pubsub_payload(event.data)

        # If there is genuinely nothing to process, just return success.
        if payload in (None, "", {}):
            logger.info({"msg": "No payload", "messageId": meta.get("pubsub_message_id")})
            return "No data"

        # Use Pub/Sub messageId for idempotency to avoid duplicate runs
        idem_key = meta.get("pubsub_message_id") or ""
        # Minimal ingestion – make idempotency visible using the Pub/Sub message id.
        _LS.create_run(
            name="pubsub-event",
            inputs={"event_text": payload},
            run_type="chain",
            project_name=_PROJECT,  # or project_id if you prefer
            metadata=meta,
        )
        return "OK"
    except Exception as e:
        # Raising triggers retries (use DLQ on the subscription to catch poison messages) 
        logger.warning({"msg": "Failed to ingest to LangSmith", "messageId": idem_key})
        raise 