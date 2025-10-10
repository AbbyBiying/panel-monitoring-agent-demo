from __future__ import annotations

import json
from datetime import datetime, UTC
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from google.cloud import firestore
from langsmith import traceable

from panel_monitoring.app.schemas import Signals, ModelMeta
from panel_monitoring.app.utils import looks_like_automated
from panel_monitoring.app.clients.llms.provider_base import ClassifierProvider
from panel_monitoring.data.firestore_client import events_col, runs_col

def _utcnow() -> datetime:
    return datetime.now(UTC)

def _heuristic_fallback(text: str) -> Tuple[Signals, ModelMeta]:
    t = (text or "").lower()
    flagged = any(k in t for k in ("suspicious", "fraud", "bot", "abuse", "disposable"))
    return (
        Signals(suspicious_signup=flagged, normal_signup=not flagged, confidence=0.5,
                reason="LLM unavailable/invalid; simple heuristic used."),
        ModelMeta(provider="heuristic", model="fallback"),
    )

_CLASSIFIER_PROVIDER: Optional[ClassifierProvider] = None
def set_classifier_provider(p: ClassifierProvider) -> None:
    global _CLASSIFIER_PROVIDER
    _CLASSIFIER_PROVIDER = p

@traceable(tags=["node"])
def user_event_node(state: Dict[str, Any]) -> Dict[str, Any]:
    project_id = state.get("project_id") or "panel-app-dev"

    raw_text = state.get("event_text")
    if isinstance(raw_text, str):
        event_text = raw_text
    else:
        ev_src = state.get("event_data") or ""
        event_text = ev_src if isinstance(ev_src, str) else json.dumps(ev_src, ensure_ascii=False)

    event_id: Optional[str] = state.get("event_id")
    if not event_id:
        ref = events_col().document()
        ref.set({
            "project_id": project_id,
            "type": "signup",
            "source": "web",
            "received_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
            "event_at": _utcnow(),
            "status": "pending",
            "payload": {"preview": event_text[:200]},
        })
        event_id = ref.id

    return {
        "project_id": project_id,
        "event_id": event_id,
        "event_text": event_text,
        "signals": {},
        "classification": "error",
        "action": "",
        "log_entry": "",
    }

@traceable(tags=["node"])
def signal_evaluation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    text = state.get("event_text") or ""
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)

    if looks_like_automated(text):
        signals = Signals(suspicious_signup=True, normal_signup=False, confidence=0.9,
                          reason="Automated/garbled input.")
        meta = ModelMeta(provider="heuristic", model="shortcircuit")
    else:
        try:
            if _CLASSIFIER_PROVIDER is None:
                raise RuntimeError("provider not set")
            out = (_CLASSIFIER_PROVIDER.classify(text)
                   if hasattr(_CLASSIFIER_PROVIDER, "classify") else _CLASSIFIER_PROVIDER(text))
            if isinstance(out, tuple) and len(out) == 2:
                raw_sig, raw_meta = out
            else:
                raw_sig, raw_meta = out or {}, {}
            signals = Signals.model_validate(raw_sig)
            meta = ModelMeta.model_validate(raw_meta or {})
        except Exception:
            signals, meta = _heuristic_fallback(text)

    classification = "suspicious" if (signals.suspicious_signup and not signals.normal_signup) else "normal"
    action = "hold_account" if classification == "suspicious" else "no_action"

    return {
        "signals": signals.model_dump(),
        "classification": classification,
        "confidence": signals.confidence,
        "action": action,
        "model_meta": meta.model_dump(),
    }

@traceable(tags=["node"])
def action_decision_node(state: Dict[str, Any]) -> Dict[str, Any]:
    return {"action": "hold_account" if state.get("classification") == "suspicious" else "no_action"}

@traceable(tags=["node"])
def explanation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    c = state.get("classification")
    s = state.get("signals") or {}
    a = state.get("action") or "no_action"
    conf = s.get("confidence")
    conf_txt = f"{float(conf):.2f}" if isinstance(conf, (int, float)) else "N/A"
    if c == "suspicious":
        ex = f"SUSPICIOUS (Conf {conf_txt}). Reason: {s.get('reason', 'n/a')}. Action: {a}."
    elif c == "normal":
        ex = f"NORMAL (Conf {conf_txt}). Action: {a}."
    else:
        ex = "Classification error."
    return {"explanation_report": ex}

@traceable(tags=["node"])
def logging_node(state: Dict[str, Any]) -> Dict[str, Any]:
    project_id = state.get("project_id") or "panel-app-dev"
    event_id = state.get("event_id")
    if not event_id:
        return {"error": "logging_node: missing event_id"}

    run_id = uuid4().hex
    decision = state.get("classification", "error")
    confidence = float(state.get("confidence") or 0.0)
    signals = state.get("signals") or {}
    meta = state.get("model_meta") or {}

    runs_col().document(run_id).set({
        "project_id": project_id,
        "event_id": event_id,
        "provider": meta.get("provider", "vertexai"),
        "model": meta.get("model", "gemini-2.5-pro"),
        "decision": decision,
        "confidence": confidence,
        "signals": signals,
        "started_at": firestore.SERVER_TIMESTAMP,
        "finished_at": firestore.SERVER_TIMESTAMP,
    })

    events_col().document(event_id).set({
        "project_id": project_id,
        "status": "classified" if decision != "error" else "error",
        "decision": decision,
        "confidence": confidence,
        "last_run_id": run_id,
        "updated_at": firestore.SERVER_TIMESTAMP,
    }, merge=True)

    preview_src = state.get("event_text") or state.get("event_data") or "N/A"
    preview = (preview_src if isinstance(preview_src, str) else json.dumps(preview_src))[:50]
    log_summary = {
        "event_id": event_id,
        "run_id": run_id,
        "classification": decision,
        "confidence": confidence,
        "final_action": state.get("action") or ("no_action" if decision == "normal" else "N/A"),
        "provider": meta.get("provider", "vertexai"),
        "model": meta.get("model", "gemini-2.5-pro"),
        "event_preview": f"{preview}...",
        "timestamp": _utcnow().isoformat(),
    }
    return {"log_entry": json.dumps(log_summary, indent=2)}
