# test_local.py
import base64
import datetime as dt
from cloudevents.http import CloudEvent

from gcp.functions.pubsub_to_langsmith.main import pubsub_to_langsmith

import gcp.functions.pubsub_to_langsmith.main as mod

def make_event(payload_text: str):
    attrs = {
        "type": "google.cloud.pubsub.topic.v1.messagePublished",
        "source": "//pubsub.googleapis.com/projects/demo/topics/user-event-signups",
        "id": "local-test-1",
        "specversion": "1.0",
        "time": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
    }
    data = {
        "message": {
            "data": base64.b64encode(payload_text.encode("utf-8")).decode("ascii"),
            "messageId": "16668885052829097",
            "orderingKey": None,
            "publishTime": "2025-10-22T00:04:31.696Z",
            "attributes": {"source": "pytest"},
        },
        "subscription": "projects/demo/subscriptions/local-sub",
    }
    return CloudEvent(attrs, data)

# --- Minimal stub so we don't call the real deployment ---
class _FakeRemoteGraph:
    def invoke(self, inputs, config=None):
        # Return something dict-like so your concise result logging prints keys
        return {"ok": True, "echo_keys": list(inputs.keys()), "thread_id": (config or {}).get("thread_id")}


if __name__ == "__main__":
    # Patch just one helper to avoid real network/auth; everything else unchanged
    mod._get_remote_graph = lambda: _FakeRemoteGraph()
    ev = make_event("this is great!!finally!LangSmith")
    print(pubsub_to_langsmith(ev))