# # Cloud Run Functions (Gen2 on Cloud Run) – Pub/Sub → LangSmith
# # Deploy:
# # gcloud functions deploy pubsub-to-langsmith \
# #   --region=us-central1 \
# #   --runtime=python313 \
# #   --source=. \
# #   --entry-point=pubsub_to_langsmith \
# #   --trigger-topic=user-event-signups

# import os
# import base64
# import typing as t
# from dotenv import load_dotenv

# import functions_framework
# from langsmith import Client

# load_dotenv()

# # Configure LangSmith client from env (prefer Secret Manager for the API key)
# _LS = Client(
#     api_key=os.getenv("LANGSMITH_API_KEY_CLOUD_FUNCTIONS"),
#     api_url=os.getenv("LANGSMITH_ENDPOINT"),  # optional
# )
# _PROJECT = os.getenv("LANGSMITH_PROJECT")


# def _decode_pubsub_payload(event_data: dict) -> t.Tuple[dict, dict]:
#     """
#     Extracts and decodes Pub/Sub payload + useful metadata from an Eventarc CloudEvent.

#     Eventarc (Pub/Sub) CloudEvent data shape:
#     {
#       "message": {
#         "data": "base64-encoded",
#         "messageId": "123",
#         "orderingKey": "key-1",
#         "publishTime": "2025-01-01T00:00:00Z",
#         "attributes": { ... }   # optional
#       },
#       "subscription": "projects/.../subscriptions/..."
#     }
#     """
#     msg = (event_data or {}).get("message") or {}
#     data_b64 = msg.get("data")
#     payload = {}
#     if data_b64:
#         try:
#             payload = base64.b64decode(data_b64).decode("utf-8")
#         except Exception as e:
#             print(f"Error decoding base64 data: {e}")
#             raise

#     meta = {
#         "pubsub_message_id": msg.get("messageId"),
#         "pubsub_ordering_key": msg.get("orderingKey"),
#         "pubsub_publish_time": msg.get("publishTime"),
#         "pubsub_attributes": msg.get("attributes") or {},
#     }
#     return payload, meta


# @functions_framework.cloud_event
# def pubsub_to_langsmith(event):
#     """
#     Cloud Run Function entry point for Pub/Sub.
#     - Uses CloudEvents signature (required by Eventarc).
#     - Raises on failure so Eventarc/Pub/Sub will retry per policy.
#     """
#     try:
#         print(event)
#         payload, meta = _decode_pubsub_payload(event.data)

#         # If there is genuinely nothing to process, just return success.
#         if not payload:
#             return "No data"

#         # Minimal ingestion – make idempotency visible using the Pub/Sub message id.
#         _LS.create_run(
#             name="pubsub-event",
#             inputs={"event_text": payload},
#             run_type="chain",
#             project_name=_PROJECT,  # or project_id if you prefer
#             metadata=meta,
#         )
#         return "OK"
#     except Exception as e:
#         # Raising triggers retries (use DLQ on the subscription to catch poison messages)
#         raise
# main.py
import os
import json
import base64
import typing as t

from dotenv import load_dotenv
import functions_framework
from langsmith import Client

# Local dev convenience; in Cloud Functions, envs should come from config/Secrets
load_dotenv()

# Prefer Secret Manager -> env var injection at deploy time
_LS = Client(
    api_key=os.getenv("LANGSMITH_API_KEY_CLOUD_FUNCTIONS") or os.getenv("LANGSMITH_API_KEY"),
    api_url=os.getenv("LANGSMITH_ENDPOINT"),  # optional
)
_PROJECT = os.getenv("LANGSMITH_PROJECT")  # e.g., "PubSub Ingest"

def _decode_pubsub_payload(event_data: dict) -> t.Tuple[t.Any, dict]:
    """
    Extracts and decodes Pub/Sub payload + useful metadata from an Eventarc CloudEvent.

    Expected data shape:
    {
      "message": {
        "data": "base64-encoded",
        "messageId": "123",
        "orderingKey": "key-1",
        "publishTime": "2025-01-01T00:00:00Z",
        "attributes": { ... }
      },
      "subscription": "projects/.../subscriptions/..."
    }
    """
    msg = (event_data or {}).get("message") or {}
    data_b64 = msg.get("data")
    payload: t.Any = None

    if data_b64:
        raw = base64.b64decode(data_b64).decode("utf-8")
        # Try JSON first; if not JSON, keep as text
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = raw

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
    Gen2 Cloud Functions entry point for Pub/Sub via Eventarc.
    Raise on failure so Pub/Sub retries (configure a DLQ on the subscription).
    """
    try:
        payload, meta = _decode_pubsub_payload(event.data)

        if payload in (None, "", {}):
            # Nothing to do; acknowledge without error
            print({"msg": "No payload to process", **meta})
            return "No data"

        # Minimal ingestion to LangSmith; adjust run_type/name as you like
        _LS.create_run(
            name="pubsub-event",
            inputs={"event": payload},
            run_type="chain",
            project_name=_PROJECT,
            metadata=meta,
        )
        print({"msg": "Ingested to LangSmith", **meta})
        return "OK"
    except Exception as e:
        # Let the platform retry; use a DLQ to capture poison messages
        # (Optional) add more context here
        print({"error": str(e)})
        raise
