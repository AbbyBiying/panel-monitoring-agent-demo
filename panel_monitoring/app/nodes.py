# panel_monitoring/app/nodes.py
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from typing import Dict, Any
from uuid import uuid4

from google.cloud import firestore
from langsmith import traceable

from panel_monitoring.app.schemas import GraphState, ModelMeta
from panel_monitoring.app.utils import looks_like_automated
from panel_monitoring.app.clients.llms.provider_base import ClassifierProvider
from panel_monitoring.data.firestore_client import (
    events_col,   # top-level 'events'
    runs_col,     # top-level 'runs'
)

# ----------------------------
# Helpers
# ----------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _heuristic_fallback(event: str) -> Dict[str, Any]:
    ev = (event or "").lower()
    heur = any(k in ev for k in ("suspicious", "fraud", "bot", "abuse", "disposable"))
    return {
        "suspicious_signup": heur,
        "normal_signup": not heur,
        "confidence": 0.5,
        "reason": "LLM failed, used heuristic fallback.",
    }


# ----------------------------
# Nodes
# ----------------------------

@traceable(tags=["node"])
def user_event_node(state: GraphState) -> GraphState:
    """
    Create (or ensure) the parent event document in Firestore and prime state.
    If an event already exists (event_id provided), we don't create a new one.
    """
    project_id = state.get("project_id") or "panel-app-dev"
    event_text = state.get("event_text") or (state.get("event_data") or "")

    # If caller already set event_id, just pass-through.
    event_id = state.get("event_id")
    if not event_id:
        # Create minimal event doc in TOP-LEVEL 'events'
        evt_ref = events_col().document()
        evt_ref.set(
            {
                "project_id": project_id,
                "type": "signup",  # derive from input if you have a parser
                "source": "web",
                "received_at": firestore.SERVER_TIMESTAMP,
                "event_at": _utcnow(),
                "status": "pending",
                "payload": {"preview": (event_text or "")[:200]},
            }
        )
        event_id = evt_ref.id

    # Prime graph state
    return {
        "project_id": project_id,
        "event_id": event_id,
        "event_text": event_text,
        "signals": {},
        "classification": "error",
        "action": "",
        "log_entry": "",
    }


def make_signal_eval_node(provider: ClassifierProvider):
    """
    Run the classifier provider and compute (classification, confidence).
    Provider is expected to return (signals: dict, meta: dict).
    """

    @traceable(tags=["node"])
    def signal_evaluation_node(state: GraphState) -> GraphState:
        event_text = state.get("event_text", "") or ""
        model_meta: ModelMeta = {}

        # Short-circuit obvious automated noise
        if looks_like_automated(event_text):
            signals = {
                "suspicious_signup": True,
                "normal_signup": False,
                "confidence": 0.9,
                "reason": "Automated/garbled input.",
            }
            model_meta = {"provider": "heuristic", "model": "builtin-shortcircuit"}
        else:
            try:
                # Provider returns (signals, meta)
                signals, meta = (
                    provider.classify(event_text)
                    if hasattr(provider, "classify")
                    else provider(event_text)
                )
                model_meta = meta or {}
            except Exception:
                signals = _heuristic_fallback(event_text)
                model_meta = {"provider": "heuristic", "model": "fallback"}

        suspicious = bool(signals.get("suspicious_signup"))
        normal = bool(signals.get("normal_signup"))
        classification = "suspicious" if suspicious and not normal else "normal"
        confidence = float(signals.get("confidence") or 0.0)
        action = "no_action" if classification == "normal" else ""

        return {
            "signals": signals,
            "classification": classification,
            "confidence": confidence,
            "action": action,
            "model_meta": model_meta,
        }

    return signal_evaluation_node


@traceable(tags=["node"])
def action_decision_node(state: GraphState) -> GraphState:
    """
    Placeholder decision policy. Replace with your real policy rules.
    """
    if state.get("classification") == "suspicious":
        action = "remove_account" if random.random() > 0.5 else "hold_account"
    else:
        action = "no_action"
    return {"action": action}


@traceable(tags=["node"])
def explanation_node(state: GraphState) -> GraphState:
    c = state.get("classification")
    s = state.get("signals", {}) or {}
    a = state.get("action") or "no_action"
    conf = s.get("confidence")
    conf_txt = f"{conf:.2f}" if isinstance(conf, (int, float)) else "N/A"

    if c == "suspicious":
        ex = (
            f"The event was flagged as **SUSPICIOUS** (Confidence: {conf_txt}).\n"
            f"Primary concern: **{s.get('reason', 'n/a')}**\n"
            f"Final action: **{a}**."
        )
    elif c == "normal":
        ex = (
            f"The event was classified as **NORMAL** (Confidence: {conf_txt}). "
            f"No sufficient evidence of fraud; action: **{a}**."
        )
    else:
        ex = "Classification error occurred; see logs."
    return {"explanation_report": ex}


@traceable(tags=["node"])
def logging_node(state: GraphState) -> GraphState:
    """
    Persist a run (top-level 'runs') and update the parent event summary.
    Metrics rollups removed for simplicity.
    """
    project_id = state["project_id"]
    event_id = state["event_id"]
    run_id = uuid4().hex

    decision = state.get("classification", "error")
    confidence = float(state.get("confidence") or 0.0)
    signals = state.get("signals", {}) or {}
    meta: ModelMeta = state.get("model_meta", {}) or {}

    # ---- Write run (TOP-LEVEL 'runs')
    runs_col().document(run_id).set(
        {
            "project_id": project_id,
            "event_id": event_id,

            "provider": meta.get("provider", "vertexai"),
            "model": meta.get("model", "gemini-2.5-pro"),
            "temperature": meta.get("temperature", 0),
            "max_output_tokens": meta.get("max_output_tokens"),
            "request_timeout": meta.get("request_timeout"),
            "max_retries": meta.get("max_retries"),
            "usage": meta.get("usage", {}),
            "latency_ms": meta.get("latency_ms"),
            "cost_usd": meta.get("cost_usd"),

            "event_type": "signup",  # include if you want quick filtering in runs
            "decision": decision,
            "confidence": confidence,
            "signals": signals,

            "started_at": firestore.SERVER_TIMESTAMP,
            "finished_at": firestore.SERVER_TIMESTAMP,
            "error": meta.get("error"),
        }
    )

    # ---- Update parent event summary (TOP-LEVEL 'events')
    events_col().document(event_id).set(
        {
            "project_id": project_id,
            "status": "classified" if decision != "error" else "error",
            "decision": decision,
            "confidence": confidence,
            "last_run_id": run_id,
            "updated_at": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )

    # ---- Build a small human-friendly log summary
    action = state.get("action") or ("no_action" if decision == "normal" else "N/A")
    preview = (state.get("event_text") or state.get("event_data") or "N/A")[:50]
    log_summary = {
        "event_id": event_id,
        "run_id": run_id,
        "event_preview": f"{preview}...",
        "classification": decision,
        "confidence": confidence,
        "final_action": action,
        "provider": meta.get("provider", "vertexai"),
        "model": meta.get("model", "gemini-2.5-pro"),
        "latency_ms": meta.get("latency_ms"),
        "timestamp": _utcnow().isoformat(),
    }

    return {"log_entry": json.dumps(log_summary, indent=2)}
