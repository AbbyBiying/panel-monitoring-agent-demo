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


def _init_remote_graph() -> None:
    """
    Synchronous startup initializer — call at module level to exploit
    Cloud Run Gen2's CPU-boosted warm-up phase.
    Required env:
      - LG_DEPLOYMENT_URL: https://<your-deployment-host>
      - Either LG_ASSISTANT_ID or LG_GRAPH_NAME
      - LANGSMITH_API_KEY (auth for RemoteGraph)
    """
    global _remote_graph

    if _remote_graph is not None:
        return

    cf_key = os.getenv("LANGSMITH_API_KEY_CLOUD_FUNCTIONS")
    if cf_key and not os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = cf_key

    url = os.environ["LG_DEPLOYMENT_URL"].rstrip("/")
    target = os.getenv("LG_ASSISTANT_ID") or os.getenv("LG_GRAPH_NAME", "panel_agent")
    if not target:
        raise RuntimeError("Set LG_ASSISTANT_ID or LG_GRAPH_NAME in env.")

    logger.info("Initializing RemoteGraph: target=%s url=%s", target, url)
    _remote_graph = RemoteGraph(target, url=url)


def _get_remote_graph() -> RemoteGraph:
    """Returns the pre-initialized singleton. Fails loud if startup was skipped."""
    if _remote_graph is None:
        raise RuntimeError(
            "RemoteGraph not initialized. _init_remote_graph() must run at module load."
        )
    return _remote_graph


def _build_graph_inputs(payload: t.Union[dict, str, None], meta: dict) -> dict:
    """Shape Pub/Sub payload to match GraphState schema (from your nodes.py)."""
    base = {"project_id": "panel-app-dev", "meta": meta}
    if isinstance(payload, str):
        base["event_text"] = payload
    elif isinstance(payload, dict):
        base["event_data"] = payload
    else:
        base["event_text"] = ""
    return base


def _runtime_config(thread_id: str) -> dict:
    return {
        "thread_id": thread_id,
        "project_name": os.getenv("LANGSMITH_PROJECT", "panel-monitoring-agent"),
        "configurable": {
            "provider": os.getenv("PANEL_DEFAULT_PROVIDER", "vertexai"),
            "model": os.getenv("VERTEX_MODEL", "gemini-2.5-pro"),
        },
    }


@functions_framework.cloud_event
async def pubsub_to_langsmith(event):
    """
    Cloud Run Function entry point.
    Triggered by Pub/Sub → sends payload to LangSmith deployed graph ("panel-agent").
    """
    # 1) Decode Pub/Sub envelope
    payload, meta = _decode_pubsub_payload(event.data)
    pubsub_message_id = (meta or {}).get("pubsub_message_id") or ""

    if payload in (None, "", {}):
        logger.info("No payload, messageId=%s", pubsub_message_id)
        return "No data"

    # 2) Build inputs for your graph
    inputs = _build_graph_inputs(payload, meta)

    attrs = (meta or {}).get("pubsub_attributes") or {}
    thread_id = (
        attrs.get("thread_id")
        or pubsub_message_id
        or str(uuid.uuid4())
    )

    # 3) Invoke the remote graph (singleton, reuses HTTP pool)
    remote = _get_remote_graph()
    try:
        config = _runtime_config(thread_id)
        result = await asyncio.wait_for(
            remote.ainvoke(inputs, config=config),
            timeout=float(os.getenv("GRAPH_INVOKE_TIMEOUT_SECS", "300")),
        )
        logger.info("invoke_ok message_id=%s thread_id=%s", pubsub_message_id, thread_id)

        return {
            "ok": True,
            "message_id": pubsub_message_id,
            "thread_id": thread_id,
            "result": result,
        }
    except asyncio.TimeoutError:
        logger.error("Graph invocation timed out: message_id=%s thread_id=%s",
                      pubsub_message_id, thread_id)
        raise
    except Exception as e:
        logger.error("Graph invocation failed: %s", e, extra={
            "message_id": pubsub_message_id,
            "thread_id": thread_id,
        })
        raise


# ── Module-level warm-up ──────────────────────────────────────────────
# Cloud Run Gen2 provides CPU boost during container startup.
# Initialize the RemoteGraph here so the first Pub/Sub message
# doesn't pay the cold-start penalty.
# K_SERVICE is set automatically by Cloud Run; skip when running locally.
if os.getenv("K_SERVICE"):
    _init_remote_graph()