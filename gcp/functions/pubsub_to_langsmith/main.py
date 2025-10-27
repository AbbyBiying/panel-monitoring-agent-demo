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

import base64
import json
import os
import time
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

LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
print(f"LANGSMITH_PROJECT: {LANGSMITH_PROJECT}")


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
        print(f"[ERROR] Failed to decode Pub/Sub payload: {e}")
        print(f"  data_b64: {data_b64!r}")
        print(f"  msg: {msg!r}")
        print(f"  raw: {locals().get('raw', None)!r}")
        print(f"  event_data: {event_data!r}")
        raise


def _get_remote_graph() -> RemoteGraph:
    """
    Builds a RemoteGraph bound to your LangSmith Deployment.
    Required env:
      - LG_DEPLOYMENT_URL: https://<your-deployment-host>
      - Either LG_ASSISTANT_ID or LG_GRAPH_NAME
      - LANGSMITH_API_KEY (auth for RemoteGraph)
    """
    # Allow your existing key name to flow into the standard var RemoteGraph expects.
    cf_key = os.getenv("LANGSMITH_API_KEY_CLOUD_FUNCTIONS")

    if cf_key and not os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = cf_key

    url = os.environ["LG_DEPLOYMENT_URL"].rstrip("/")
    print(f"LG_DEPLOYMENT_URL: {url}")
    target = os.getenv("LG_ASSISTANT_ID") or os.getenv("LG_GRAPH_NAME", "panel_agent")
    if not target:
        raise RuntimeError("Set LG_ASSISTANT_ID or LG_GRAPH_NAME in env.")

    print(f"[RemoteGraph WITH target AND URL] target={target} url={url}")
    return RemoteGraph(target, url=url)


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
    # Pull from env so you can set them with --set-env-vars at deploy time
    return {
        "thread_id": thread_id,
        "project_name": os.getenv("LANGSMITH_PROJECT", "panel-monitoring-agent"),
        "configurable": {
            "provider": "vertexai",
            "model": "gemini-2.5-pro"
        },
    }


@functions_framework.cloud_event
def pubsub_to_langsmith(event):
    """
    Cloud Run Function entry point.
    Triggered by Pub/Sub → sends payload to LangSmith deployed graph ("panel-agent").
    """
    print("[BOOT] pubsub_to_langsmith invoked")
    # 1) Decode Pub/Sub envelope
    payload, meta = _decode_pubsub_payload(event.data)
    pubsub_message_id = (meta or {}).get("pubsub_message_id") or ""

    if payload in (None, "", {}):
        print({"msg": "No payload", "messageId": pubsub_message_id})
        return "No data"

    # 2) Build inputs for your graph
    inputs = _build_graph_inputs(payload, meta)

    attrs = (meta or {}).get("pubsub_attributes") or {}
    thread_id = (
        attrs.get("thread_id")
        or attrs.get("user_id")
        or attrs.get("ordering_key")
        or (meta or {}).get("pubsub_ordering_key")
        or pubsub_message_id
        or str(uuid.uuid4())
    )
    # 3) Invoke the remote graph
    remote = _get_remote_graph()
    print(f"remote graph obtained, invoking...{remote}:{thread_id}")
    try:
        config = _runtime_config(thread_id)
        # print out the provider being used
        print(f"Using provider: {config['configurable']['provider']}")
        result = remote.invoke(inputs, config=config)
        print(
            {
                "stage": "invoke_ok",
                "message_id": pubsub_message_id,
                "thread_id": thread_id,
            }
        )
        # Optional: log a concise result summary to avoid huge logs
        if isinstance(result, dict):
            print(f"[RESULT.keys] {list(result.keys())}")
            print(f"log_entry - {result.get('log_entry', '')[:200]!r}...")
        else:
            print(f"[RESULT.type] {type(result)}")

        time.sleep(0.2)
        print("[COMPLETE] Graph invocation finished.")
        # how to parse the result?
        print("result", result)

        return {
            "ok": True,
            "message_id": pubsub_message_id,
            "thread_id": thread_id,
            "result": result,
        }
    except Exception as e:
        # Let Eventarc decide on retries; you’ll also see errors in LangSmith traces/logs
        print(f"[ERROR] Graph invocation failed: {e}")
        raise
