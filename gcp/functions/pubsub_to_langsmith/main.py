# panel-monitoring-agent/gcp/functions/pubsub_to_langsmith/main.py

# Cloud Run Functions (Gen2 on Cloud Run) – Pub/Sub → LangSmith (RemoteGraph)
# Deploy (example):
# gcloud functions deploy pubsub-to-langsmith \
#   --gen2 \
#   --region=us-central1 \
#   --runtime=python313 \
#   --source=. \
#   --entry-point=pubsub_to_langsmith \
#   --trigger-topic=user-event-signups \
#   --set-env-vars=LANGSMITH_API_KEY=***,LG_DEPLOYMENT_URL=https://<your-deployment-host>,LG_GRAPH_NAME=<your-graph-name>,LANGSMITH_PROJECT=<your-project>

import asyncio
import base64
import json
import logging
import os
import typing as t
import uuid

import functions_framework
from dotenv import load_dotenv

# RemoteGraph is the official client facade for calling a LangGraph deployment.
# It gives you API parity with a local CompiledGraph: invoke(), stream(), get_state(), etc.
# Docs: https://docs.langchain.com/langsmith/use-remote-graph
from langgraph.pregel.remote import RemoteGraph

# Load local env only when running locally; no-op on Cloud Run/Functions
load_dotenv()

logger = logging.getLogger(__name__)


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

    Returns (payload, meta) where:
      - payload: dict if JSON, str if not JSON, or None.
      - meta:    pubsub IDs + attributes for tracing.
    Extracts and decodes Pub/Sub payload + metadata from an Eventarc CloudEvent.
    """
    msg = (event_data or {}).get("message") or {}
    data_b64 = msg.get("data")
    payload = None

    try:
        if data_b64:
            raw = base64.b64decode(data_b64).decode("utf-8")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = raw
        else:
            raw = None

        meta = {
            "pubsub_message_id": msg.get("messageId"),
            "pubsub_ordering_key": msg.get("orderingKey"),
            "pubsub_publish_time": msg.get("publishTime"),
            "pubsub_attributes": msg.get("attributes") or {},
        }
        return payload, meta

    except Exception as e:
        logger.error("Failed to decode Pub/Sub payload: %s", e)
        raise


# ── Entry Points ────────────────────────────────────────────────────────


@functions_framework.cloud_event
def pubsub_to_langsmith(cloud_event):
    """
    Sync entry point for GCP Cloud Functions.
    Bridges the Secret Manager env var to the expected SDK name.
    """

    if "LANGSMITH_API_KEY" not in os.environ:
        os.environ["LANGSMITH_API_KEY"] = os.getenv(
            "LANGSMITH_API_KEY_CLOUD_FUNCTIONS", ""
        )

    return asyncio.run(async_handler(cloud_event))


async def async_handler(cloud_event):
    """
    Main async logic: Decodes Pub/Sub and calls the remote graph.
    """
    # 1) Auth Setup
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        logger.error("LANGSMITH_API_KEY is missing. Check Secret Manager.")
        return {"ok": False, "error": "Missing API Key"}

    # 2) Data Decoding
    payload, meta = _decode_pubsub_payload(cloud_event.data)
    pubsub_message_id = (meta or {}).get("pubsub_message_id") or "unknown"

    if payload in (None, "", {}):
        logger.info("No payload found in message %s", pubsub_message_id)
        return "No data"

    # 3) Thread ID & Inputs
    inputs = _build_graph_inputs(payload, meta)
    attrs = (meta or {}).get("pubsub_attributes") or {}

    # Use existing thread_id if provided, otherwise the pubsub message id
    seed_id = attrs.get("thread_id") or pubsub_message_id

    # Deterministic UUID ensures idempotency on retries
    thread_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(seed_id)))

    # 4) Initialize RemoteGraph
    url = os.environ["LG_DEPLOYMENT_URL"].rstrip("/")
    target = os.getenv("LG_ASSISTANT_ID") or os.getenv("LG_GRAPH_NAME", "panel_agent")

    remote = RemoteGraph(target, url=url, api_key=api_key)

    try:
        config = {
            "configurable": {"thread_id": thread_id},
            "project_name": os.getenv("LANGSMITH_PROJECT", "panel-monitoring-agent"),
        }

        # 5) Invoke the Remote Graph
        result = await asyncio.wait_for(
            remote.ainvoke(inputs, config=config),
            timeout=float(os.getenv("GRAPH_INVOKE_TIMEOUT_SECS", "300")),
        )

        logger.info("Invoke success | thread_id=%s", thread_id)
        return {"ok": True, "result": result}

    except Exception as e:
        logger.error("Invoke failed: %s | thread_id=%s", e, thread_id)
        raise


def _build_graph_inputs(payload, meta) -> dict:
    """Formats the payload for the LangGraph state schema."""
    return {
        "event_data": payload if isinstance(payload, dict) else {},
        "event_text": payload if isinstance(payload, str) else "",
        "meta": meta,
    }
