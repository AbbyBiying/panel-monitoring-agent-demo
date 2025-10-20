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
import time

import functions_framework
from langsmith import Client

# Load environment variables from .env file when running locally; does nothing when deployed to GCP
from dotenv import load_dotenv
load_dotenv()

LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
print(f"LANGSMITH_PROJECT: {LANGSMITH_PROJECT}")

def _get_ls_client() -> Client:
    api_key = os.getenv("LANGSMITH_API_KEY_CLOUD_FUNCTIONS")
    if not api_key:
        # Fail fast so the error is obvious in logs; Pub/Sub will retry if enabled 
        print(
            {
                "msg": "Missing LangSmith API key in environment.",
                "checked": ["LANGSMITH_API_KEY_CLOUD_FUNCTIONS"],
            }
        )
        raise RuntimeError("Missing LANGSMITH API key")
    return Client(api_key=api_key)


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
            print(f"payload extracted: {payload}")
            meta = {
                "pubsub_message_id": msg.get("messageId"),
                "pubsub_ordering_key": msg.get("orderingKey"),
                "pubsub_publish_time": msg.get("publishTime"),
                "pubsub_attributes": msg.get("attributes") or {},
            }
            print(f"meta extracted: {meta}")
            return payload, meta
        except Exception as e:
            print(f"Error decoding base64 data: {e}")
            print(f"raw: {raw}")
            print(f"msg: {msg}")
            print(f"data_b64: {data_b64}")
            print(f"event_data: {event_data}")
            raise

@functions_framework.cloud_event
def pubsub_to_langsmith(event):
    print(f"LANGSMITH_PROJECT: {LANGSMITH_PROJECT}")
    # Configure LangSmith client from env (prefer Secret Manager for the API key)

    """
    Cloud Run Function entry point for Pub/Sub.
    - Uses CloudEvents signature (required by Eventarc).
    - Raises on failure so Eventarc/Pub/Sub will retry per policy.
    """
    try:
        print(f"event.data: {event.data}")
        payload, meta = _decode_pubsub_payload(event.data)
        pubsub_message_id = meta.get("pubsub_message_id") or ""

        # If there is genuinely nothing to process, just return success.
        if payload in (None, "", {}):
            print({"msg": "No payload", "messageId": pubsub_message_id})
            return "No data"

        _LS = _get_ls_client()

        # Minimal ingestion – make idempotency visible using the Pub/Sub message id.
        name = f"user-event-signups-{pubsub_message_id}"
        print(f"about to create run: {name}")
        _LS.create_run(
            name=name,
            inputs={"event_text": payload},
            run_type="chain",
            project_name=LANGSMITH_PROJECT,
            metadata=meta,
        )
        print(f"Run created: {name}")
        time.sleep(5)
        runs = _LS.list_runs(project_name=LANGSMITH_PROJECT)
        print(f"runs: {runs}")
        for run in runs:
            print(f"run.status: {run.status}")

        return f"Run created: {name}"
    except Exception as e:
        print(f"Failed to ingest to LangSmith: {e}")
        print(f"pubsub_message_id: {pubsub_message_id}")
        return "error"
