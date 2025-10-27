# panel_monitoring/app/nodes.py
from __future__ import annotations

import json
from datetime import datetime, UTC
from typing import Literal, Tuple
from uuid import uuid4

# from google.cloud import firestore
from langsmith import traceable

# from langchain_core.runnables.config import get_config

from panel_monitoring.app.clients.llms import get_llm_classifier
from panel_monitoring.app.schemas import GraphState, Signals, ModelMeta
from panel_monitoring.app.utils import looks_like_automated
# from panel_monitoring.data.firestore_client import events_col, runs_col


# --- helpers ---------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _heuristic_fallback(text: str) -> Tuple[Signals, ModelMeta]:
    t = (text or "").lower()
    flagged = any(k in t for k in ("suspicious", "fraud", "bot", "abuse", "disposable"))
    return (
        Signals(
            suspicious_signup=flagged,
            normal_signup=not flagged,
            confidence=0.5,
            reason="LLM unavailable/invalid; simple heuristic used.",
        ),
        ModelMeta(provider="heuristic", model="fallback"),
    )


def _event_text_from_state(state: GraphState) -> str:
    """Produce a plain-text representation for classification & preview."""
    if isinstance(state.event_text, str):
        return state.event_text
    src = state.event_data or ""
    if isinstance(src, str):
        return src
    return json.dumps(src, ensure_ascii=False)


# --- nodes -----------------------------------------------------------------


@traceable(tags=["node"])
def user_event_node(state: GraphState) -> GraphState:
    """
    Create (or ensure) the parent event document in Firestore and prime state.
    If an event already exists (event_id provided), we don't create a new one.
    """
    project_id = state.project_id or "panel-app-dev"

    event_text = _event_text_from_state(state)
    event_id = state.event_id or uuid4().hex
    # event_id = state.event_id
    # if not event_id:
    #     evt_ref = events_col().document()
    #     evt_ref.set(
    #         {
    #             "project_id": project_id,
    #             "type": "signup",  # TODO: derive from input if you have a parser
    #             "source": "web",
    #             "received_at": firestore.SERVER_TIMESTAMP,
    #             "updated_at": firestore.SERVER_TIMESTAMP,
    #             "event_at": _utcnow(),
    #             "status": "pending",
    #             "payload": {"preview": (event_text or "")[:200]},
    #         }
    #     )
    #     event_id = evt_ref.id

    # Seed a valid Signals to satisfy Pydantic until classifier overwrites it
    seeded_signals = Signals(
        suspicious_signup=False,
        normal_signup=True,
        confidence=0.0,
        reason="unclassified",
    )

    # Return updated GraphState (don’t mutate dicts; keep it typed)
    # state is a Pydantic model instance of GraphState
    # In Pydantic v2, .model_copy() replaces the old .copy() method from v1.
    # new_state = state.model_copy(update={"classification": "suspicious"})
    # same as new_state = GraphState(**state.dict(), classification="suspicious")
    # You don’t want to mutate the original state (since multiple nodes might reference it concurrently or in branches).
    # Instead, you return an immutable copy with the updated fields.
    # Old state remains unchanged.
    # New state carries your updates (e.g., classification results).
    # LangGraph passes this new state downstream.
    # Keeps all existing fields and values from state (anything you didn’t touch stays exactly the same).
    # Updates only the specified fields (classification in this case).
    # Preserves typing, defaults, and validation from the original model.
    # Does not mutate the original state — it returns a brand-new one.
    return state.model_copy(
        update={
            "project_id": project_id,
            "event_id": event_id,
            "event_text": event_text,
            "signals": seeded_signals,
            "classification": "error",
            "action": "",
            "log_entry": "",
        }
    )


@traceable(name="signal_evaluation_node", tags=["node"])
def signal_evaluation_node(state: GraphState) -> GraphState:
    """
    Evaluate signals using provider from runtime.context (if present).
    Falls back to lightweight heuristics when provider is absent or fails.
    """
    text = _event_text_from_state(state)

    # 1) Fast-path heuristic
    if looks_like_automated(text):
        signals = Signals(
            suspicious_signup=True,
            normal_signup=False,
            confidence=0.9,
            reason="Automated/garbled input.",
        )
        meta = ModelMeta(provider="heuristic", model="shortcircuit")

    else:
        try:
            # 2) Preferred: provider injected at invoke-time
            # cfg = ensure_config()
            # Always prefer vertexai for this deployment
            # provider = "vertexai"
            provider = "openai"
            # model = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")
            model = "gpt-4o-mini"

            print(f"[INFO] Using provider: {provider}, model: {model}")

            classifier = get_llm_classifier(provider)
            if classifier is None:
                raise RuntimeError("No LLM classifier found.")

            # Vertex classifier should handle (text) → (Signals, ModelMeta)
            call = getattr(classifier, "classify", classifier)
            if not callable(call):
                raise TypeError("Classifier is not callable and has no .classify()")

            out = call(text)

            # Accept either dict or (dict, dict)
            if isinstance(out, tuple) and len(out) == 2:
                raw_sig, raw_meta = out
            else:
                raw_sig, raw_meta = out or {}, {}

            signals = Signals.model_validate(raw_sig)
            meta = ModelMeta.model_validate(raw_meta or {})
            meta.provider = provider
            meta.model = model
        except Exception:
            print("[WARN] LLM classification failed, falling back to heuristic.")
            signals, meta = _heuristic_fallback(text)
            meta.provider = provider
            meta.model = model

    # Normalize outputs
    classification: Literal["suspicious", "normal"] = (
        "suspicious"
        if (signals.suspicious_signup and not signals.normal_signup)
        else "normal"
    )
    confidence = signals.confidence or 0.0
    action = "hold_account" if classification == "suspicious" else "no_action"

    # Return a new GraphState (typed), no intermediate dicts
    return state.model_copy(
        update={
            "signals": signals,  # keep as Pydantic model
            "classification": classification,
            "confidence": confidence,
            "action": action,
            "model_meta": meta,  # keep as Pydantic model
        }
    )


@traceable(tags=["node"])
def action_decision_node(state: GraphState) -> GraphState:
    action = "hold_account" if state.classification == "suspicious" else "no_action"
    return state.model_copy(update={"action": action})


@traceable(tags=["node"])
def explanation_node(state: GraphState) -> GraphState:
    cls = state.classification
    sig = state.signals
    action = state.action or "no_action"
    conf = sig.confidence if (sig and sig.confidence is not None) else None
    conf_txt = f"{float(conf):.2f}" if isinstance(conf, (int, float)) else "N/A"

    if cls == "suspicious":
        reason = sig.reason if (sig and sig.reason) else "n/a"
        ex = f"SUSPICIOUS (Conf {conf_txt}). Reason: {reason}. Action: {action}."
    elif cls == "normal":
        ex = f"NORMAL (Conf {conf_txt}). Action: {action}."
    else:
        ex = "Classification error."

    return state.model_copy(update={"explanation_report": ex})


@traceable(tags=["node"])
def logging_node(state: GraphState) -> GraphState:
    # project_id = state.project_id or "panel-app-dev"
    # event_id = state.event_id

    event_id = state.event_id or uuid4().hex
    # if not event_id:
    #     return state.model_copy(update={"error": "logging_node: missing event_id"})

    run_id = uuid4().hex
    decision = state.classification or "error"
    confidence = float(state.confidence or 0.0)

    # Convert models to dicts for Firestore I/O
    # signals_dict = state.signals.model_dump() if state.signals else {}
    meta = state.model_meta  # always a ModelMeta (has defaults)

    # Write run (top-level 'runs')
    # runs_col().document(run_id).set(
    #     {
    #         "project_id": project_id,
    #         "event_id": event_id,
    #         "provider": meta.provider or "vertexai",
    #         "model": meta.model or "gemini-2.5-pro",
    #         "decision": decision,
    #         "confidence": confidence,
    #         "signals": signals_dict,
    #         "started_at": firestore.SERVER_TIMESTAMP,
    #         "finished_at": firestore.SERVER_TIMESTAMP,
    #         "latency_ms": meta.latency_ms,
    #         "cost_usd": meta.cost_usd,
    #     }
    # )

    # Update parent event summary (top-level 'events')
    # events_col().document(event_id).set(
    #     {
    #         "project_id": project_id,
    #         "status": "classified" if decision != "error" else "error",
    #         "decision": decision,
    #         "confidence": confidence,
    #         "last_run_id": run_id,
    #         "updated_at": firestore.SERVER_TIMESTAMP,
    #     },
    #     merge=True,
    # )

    # Build human-friendly log summary
    preview = _event_text_from_state(state)[:50]

    log_summary = {
        "event_id": event_id,
        "run_id": run_id,
        "classification": decision,
        "confidence": confidence,
        "final_action": state.action
        or ("no_action" if decision == "normal" else "N/A"),
        "provider": meta.provider or "NONE",
        "model": meta.model or "NONE",
        "latency_ms": meta.latency_ms,
        "cost_usd": meta.cost_usd,
        "event_preview": f"{preview}...",
        "timestamp": _utcnow().isoformat(),
    }

    return state.model_copy(
        update={"log_entry": json.dumps(log_summary, ensure_ascii=False, indent=2)}
    )
