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
    if data_b64:
        raw = base64.b64decode(data_b64).decode("utf-8")
        try:
            payload = json.loads(raw)  # JSON first
        except json.JSONDecodeError:
            payload = raw              # else keep as text

    meta = {
        "pubsub_message_id": msg.get("messageId"),
        "pubsub_ordering_key": msg.get("orderingKey"),
        "pubsub_publish_time": msg.get("publishTime"),
        "pubsub_attributes": msg.get("attributes") or {},
    }
    return payload, meta


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
        os.environ["LANGSMITH_API_KEY"] = cf_key  # used implicitly by RemoteGraph

    url = os.environ["LG_DEPLOYMENT_URL"]
    target = os.getenv("LG_ASSISTANT_ID") or os.getenv("LG_GRAPH_NAME")
    if not target:
        raise RuntimeError("Set LG_ASSISTANT_ID or LG_GRAPH_NAME in env.")
    # RemoteGraph mirrors local graph API (invoke, stream, get_state, …)
    return RemoteGraph(target, url=url)


@functions_framework.cloud_event
def pubsub_to_langsmith(event):
    """
    Gen2 Cloud Functions entry point. Invokes your LangGraph **deployment** on LangSmith.

    Flow:
      1) Decode Pub/Sub message (CloudEvent)
      2) Derive a thread_id (stable if provided; otherwise random)
      3) Call deployment via RemoteGraph.invoke(inputs, config={"configurable":{"thread_id": ...}})
      4) ACK (return 200) with a short payload
    """
    print(f"[BOOT] LANGSMITH_PROJECT={LANGSMITH_PROJECT}")
    try:
        print(f"event.data: {event.data}")
        payload, meta = _decode_pubsub_payload(event.data)
        pubsub_message_id = (meta or {}).get("pubsub_message_id") or ""

        # If there is genuinely nothing to process, just return success.
        if payload in (None, "", {}):
            print({"msg": "No payload", "messageId": pubsub_message_id})
            return "No data"

        # Choose a stable thread id if you want persistent state across related events.
        attrs = (meta or {}).get("pubsub_attributes", {}) or {}
        thread_id = (
            attrs.get("thread_id")
            or attrs.get("user_id")
            or attrs.get("ordering_key")
            or pubsub_message_id
            or str(uuid.uuid4())
        )

        # Build the inputs your graph expects.
        # Adjust this to match your graph’s input schema.
        graph_inputs = {
            "messages": [
                {"role": "user", "content": f"event: {payload}"},
            ],
            "meta": meta,
            # If your graph expects different fields, map them here.
        }

        # Pass thread id using the config “configurable” channel (per docs)
        config = {"configurable": {"thread_id": thread_id}}
        remote = _get_remote_graph()

        print({"stage": "invoke_start", "message_id": pubsub_message_id, "thread_id": thread_id})
        result = remote.invoke(graph_inputs, config=config)  # non-streaming for background jobs
        print({"stage": "invoke_ok", "message_id": pubsub_message_id, "thread_id": thread_id})

        # Optional: tiny sleep to keep logs ordered in short-lived containers
        time.sleep(0.2)

        # Return a small response (Functions Gen2 will JSON-serialize dicts)
        return {
            "ok": True,
            "message_id": pubsub_message_id,
            "thread_id": thread_id,
            "result": result,
        }

    except Exception as e:
        # Log enough context for easy debugging; ACK with error payload (or rethrow to retry)
        print(
            json.dumps(
                {
                    "stage": "invoke_error",
                    "error": str(e),
                    "message_id": locals().get("pubsub_message_id", ""),
                }
            )
        )
        # If you want Pub/Sub retries, raise; otherwise return 200 with error info.
        # raise
        return {"ok": False, "error": str(e), "message_id": locals().get("pubsub_message_id", "")}
