# panel_monitoring/data/ingest.py

from __future__ import annotations
import hashlib
import json
import time
from google.cloud import firestore
from .firestore_client import events_col, metrics_daily_doc

def _event_id_for(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

def upsert_event(project_id: str, source_id: str, masked_payload: dict, meta: dict):
    eid = _event_id_for(masked_payload)
    ref = events_col(project_id).document(eid)
    snap = ref.get()
    if not snap.exists:
        ref.set({
            "projectId": project_id,
            "sourceId": source_id,
            "receivedAt": firestore.SERVER_TIMESTAMP,
            "contentRaw": masked_payload.get("content", ""),
            "piiMasked": True,
            "contentHashSha256": eid,
            "meta": meta or {},
            "finalRunId": None,
            "finalLabel": None,
            "finalConfidence": None,
        })
    return ref, eid

@firestore.transactional
def _add_run_txn(txn: firestore.Transaction, event_ref, run_doc: dict):
    runs = event_ref.collection("runs")
    q = runs.order_by("attemptRank", direction=firestore.Query.DESCENDING).limit(1)
    last = list(q.stream(transaction=txn))
    attempt = (last[0].get("attemptRank") if last else 0) + 1
    run_doc["attemptRank"] = attempt
    run_ref = runs.document()
    txn.set(run_ref, run_doc)
    return run_ref, attempt

def add_run(event_ref, provider_key: str, model_name: str, stats: dict):
    txn = event_ref._client.transaction()
    run_doc = {
        "status": stats.get("status", "success"),
        "providerKey": provider_key,
        "modelName": model_name,
        "startedAt": firestore.SERVER_TIMESTAMP,
        "finishedAt": None,
        "latencyMs": None,
        "promptHashSha256": stats.get("prompt_hash"),
        "inputCount": int(stats.get("input_count", 0)),
        "outputCount": int(stats.get("output_count", 0)),
        "inputCostUsd": float(stats.get("input_cost_usd", 0.0)),
        "outputCostUsd": float(stats.get("output_cost_usd", 0.0)),
        "totalCostUsd": float(stats.get("total_cost_usd", 0.0)),
        "request": stats.get("request"),
        "response": stats.get("response"),
        "meta": stats.get("meta", {}),
    }
    return _add_run_txn(txn, event_ref, run_doc)

def finalize_event(project_id: str, event_ref, run_ref, label: str, confidence: float, latency_ms: int, provider_key: str, cost_usd: float):
    run_ref.update({"finishedAt": firestore.SERVER_TIMESTAMP, "latencyMs": int(latency_ms)})
    event_ref.update({"finalRunId": run_ref.id, "finalLabel": label, "finalConfidence": float(confidence)})

    day = time.strftime("%Y-%m-%d")
    metrics_ref = metrics_daily_doc(project_id, day)
    metrics_ref.set({
        "runCount": firestore.Increment(1),
        "successCount": firestore.Increment(1),
        f"costByProvider.{provider_key}": firestore.Increment(float(cost_usd)),
        f"labelDist.{label}": firestore.Increment(1),
    }, merge=True)
