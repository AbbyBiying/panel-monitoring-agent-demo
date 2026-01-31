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

LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
logger.info("LANGSMITH_PROJECT: %s", LANGSMITH_PROJECT)

# ── RemoteGraph singleton ──────────────────────────────────────────────
# Cloud Run Gen2 instances handle concurrent requests. Reusing one
# RemoteGraph (and its underlying HTTP pool) avoids per-request TCP
# handshake overhead — critical under high Pub/Sub throughput.
_remote_graph: t.Optional[RemoteGraph] = None


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
        logger.error("Failed to decode Pub/Sub payload: %s", e, extra={
            "data_b64": data_b64,
            "msg": msg,
            "raw": locals().get("raw"),
            "event_data": event_data,
        })
        raise

 
@functions_framework.cloud_event
def pubsub_to_langsmith(cloud_event):
    """
    Sync entry point for Cloud Functions. 
    asyncio.run() creates a fresh loop for every request.
    """
    return asyncio.run(async_handler(cloud_event))

async def async_handler(cloud_event):
    """
    Main logic: Decodes Pub/Sub and calls the remote graph.
    """
    # 1) Decode Pub/Sub payload
    payload, meta = _decode_pubsub_payload(cloud_event.data)
    pubsub_message_id = (meta or {}).get("pubsub_message_id") or "unknown"

    if payload in (None, "", {}):
        logger.info("No payload found in message %s", pubsub_message_id)
        return "No data"

    # 2) Prep inputs and thread_id
    inputs = _build_graph_inputs(payload, meta)
    thread_id = (meta.get("pubsub_attributes", {}).get("thread_id") 
                 or pubsub_message_id)
    if "LANGSMITH_API_KEY" not in os.environ:
        os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY_CLOUD_FUNCTIONS", "")
    # 3) Initialize RemoteGraph INSIDE the async handler
    # This prevents the "Event loop is closed" error.
    url = os.environ["LG_DEPLOYMENT_URL"].rstrip("/")
    target = os.getenv("LG_ASSISTANT_ID") or os.getenv("LG_GRAPH_NAME", "panel_agent")
    
    # Client is created fresh for this request's event loop
    remote = RemoteGraph(target, url=url)

    try:
        config = {
            "configurable": {"thread_id": thread_id},
            "project_name": os.getenv("LANGSMITH_PROJECT", "panel-monitoring-agent")
        }

        # 4) Invoke the Remote Graph
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
    return {
        "event_data": payload if isinstance(payload, dict) else {},
        "event_text": payload if isinstance(payload, str) else "",
        "meta": meta
    }