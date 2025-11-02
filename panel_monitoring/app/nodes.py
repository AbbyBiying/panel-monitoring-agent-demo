# panel_monitoring/app/nodes.py
from __future__ import annotations

import json
from datetime import datetime, UTC
import os
from typing import Literal, Tuple
from uuid import uuid4

# from google.cloud import firestore
from langsmith import traceable
from langgraph.types import Command, interrupt

import logging

from panel_monitoring.app.clients.llms import get_llm_classifier
from panel_monitoring.app.schemas import GraphState, Signals, ModelMeta
from panel_monitoring.app.utils import looks_like_automated
# from panel_monitoring.data.firestore_client import events_col, runs_col

# --------------------------------------------------------------------
# Confidence Thresholds for Automated Decision Making
# --------------------------------------------------------------------

# If confidence is ≥ this value → auto-approve the model’s action (no human needed)
CONFIDENCE_AUTO_APPROVE_THRESHOLD = 0.90

# If confidence is ≥ this value → require Human-In-The-Loop (HITL) review
CONFIDENCE_REVIEW_THRESHOLD = 0.70

# If confidence is between 0.30 and 0.69 → model is uncertain, hold/no-action
CONFIDENCE_UNCERTAIN_THRESHOLD = 0.30

# If confidence is < this value → treat as invalid, low-quality, or fallback required
CONFIDENCE_LOW_THRESHOLD = 0.30
logger = logging.getLogger(__name__)


# --- helpers ---------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(UTC)

def _append_report(state: GraphState, line: str) -> str:
    return f"{state.explanation_report}\n{line}" if state.explanation_report else line

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
@traceable(name="perform_effects_node", tags=["node"])
def perform_effects_node(state: GraphState) -> GraphState:
    combined = state.explanation_report  # start with the existing report

    try:
        if state.action == "delete_account":
            logger.info("Deleting account for event_id=%s", state.event_id)
            combined = _append_report(state, "[INFO] effect:delete_account executed")
            return state.model_copy(update={"explanation_report": combined})

    except Exception as e:
        logger.warning("Effect execution failed.")
        line = f"[ERROR] effect_failed:{type(e).__name__} {e}"
        combined = _append_report(state, line)
        return state.model_copy(
            update={
                "explanation_report": combined,
                "action": "hold_account",
            }
        )

    # If no exception and no delete was required, just return state
    return state


@traceable(name="user_event_node", tags=["node"])
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
    #             "type": "signup",
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
            "classification": "pending",
            "action": "",
            "log_entry": "",
        }
    )


@traceable(name="signal_evaluation_node", tags=["node"])
def signal_evaluation_node(state: GraphState) -> GraphState:
    text = _event_text_from_state(state)

    if looks_like_automated(text):
        signals = Signals(
            suspicious_signup=True,
            normal_signup=False,
            confidence=0.9,
            reason="Automated/garbled input.",
        )
        meta = ModelMeta(provider="heuristic", model="shortcircuit")
    else:
        provider = os.getenv("PANEL_DEFAULT_PROVIDER", "vertexai")
        model = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")
        logger.info("Using provider=%s model=%s", provider, model)

        try:
            classifier = get_llm_classifier(provider)
            if classifier is None:
                raise RuntimeError(f"No LLM classifier found for provider='{provider}'")

            call = getattr(classifier, "classify", classifier)
            if not callable(call):
                raise TypeError("Classifier is not callable and has no .classify()")

            out = call(text)
            logger.debug("[DBG] LLM raw out: %r", out)

            if isinstance(out, tuple) and len(out) == 2:
                raw_sig, raw_meta = out
            else:
                raw_sig, raw_meta = (out or {}), {}

            # normalize a few common variants
            label = (
                raw_sig.get("label") or raw_sig.get("classification") or ""
            ).lower()
            if "suspicious_signup" not in raw_sig and label:
                raw_sig["suspicious_signup"] = label == "suspicious"
            if "normal_signup" not in raw_sig:
                raw_sig["normal_signup"] = not raw_sig.get("suspicious_signup", False)
            if "confidence" not in raw_sig:
                raw_sig["confidence"] = (
                    raw_sig.get("score")
                    or raw_sig.get("prob")
                    or raw_meta.get("confidence", 0.0)
                )
            if "reason" not in raw_sig:
                raw_sig["reason"] = raw_sig.get("message") or raw_meta.get("reason")

            signals = Signals.model_validate(raw_sig)
            meta = ModelMeta.model_validate(raw_meta or {})
            meta.provider = provider
            meta.model = model
            print("[DBG] normalized signals:", signals)
            print("[DBG] meta:", meta)

        except Exception as e:
            logger.warning("LLM classification failed; using heuristic fallback.")
            signals, meta = _heuristic_fallback(text)
            meta.provider = provider
            meta.model = model
            signals.reason = f"{signals.reason} ({type(e).__name__})"

    classification: Literal["suspicious", "normal"] = (
        "suspicious"
        if (signals.suspicious_signup and not signals.normal_signup)
        else "normal"
    )
    confidence = signals.confidence or 0.0
    action = "hold_account" if classification == "suspicious" else "no_action"

    return state.model_copy(
        update={
            "signals": signals,
            "classification": classification,
            "confidence": confidence,
            "action": action,
            "model_meta": meta,
        }
    )


@traceable(name="action_decision_node", tags=["node"])
def action_decision_node(state: GraphState) -> GraphState:
    """
    Safety-first policy:
      - suspicious & confidence >= CONFIDENCE_REVIEW_THRESHOLD -> request_human_review
      - suspicious & confidence <  CONFIDENCE_REVIEW_THRESHOLD -> hold_account
      - normal -> no_action
    """
    conf = float(state.confidence or 0.0)

    if state.classification == "suspicious" and conf >= CONFIDENCE_REVIEW_THRESHOLD:
        action = "request_human_review"
    elif state.classification == "suspicious":
        action = "hold_account"
    else:
        action = "no_action"

    return state.model_copy(update={"action": action})


@traceable(name="explanation_node", tags=["node"])
def explanation_node(state: GraphState) -> GraphState:
    cls = state.classification
    sig = state.signals
    action = state.action or "no_action"

    # safe confidence string
    conf_val = (sig.confidence if (sig and sig.confidence is not None) else state.confidence)
    conf_txt = f"{float(conf_val):.2f}" if isinstance(conf_val, (int, float)) else "N/A"

    # human-friendly one-liner ("ex")
    if cls == "suspicious":
        reason = sig.reason if (sig and sig.reason) else "n/a"
        ex = f"SUSPICIOUS (Conf {conf_txt}). Reason: {reason}. Action: {action}."
    elif cls == "normal":
        ex = f"NORMAL (Conf {conf_txt}). Action: {action}."
    else:
        ex = "Classification error."

    if action == "request_human_review" and getattr(state, "review_url", None):
        ex = f"{ex} Pending human review → {state.review_url}"

    combined = _append_report(state, ex)
    return state.model_copy(update={"explanation_report": combined})


@traceable(name="human_approval_node", tags=["node"])
def human_approval_node(state: GraphState) -> GraphState | Command[Literal["explain"]]:
    """
    If we requested human review, PAUSE here and ask an admin.
    When resumed, convert the review decision into the final action.
    """
    if state.action != "request_human_review":
        # Not a gated path; continue downstream
        return state

    preview = (state.event_text or "")[:200]
    prompt = {
        "title": "Human review required",
        "event_id": state.event_id,
        "classification": state.classification,
        "confidence": state.confidence,
        "signals": state.signals.model_dump() if state.signals else {},
        "provider": (state.model_meta.provider if state.model_meta else None),
        "model": (state.model_meta.model if state.model_meta else None),
        "event_preview": preview,
        "question": "Approve deletion? Reply: 'approve' | 'reject' | 'escalate'",
    }

    decision = interrupt(prompt)  # pauses run until resumed

    if isinstance(decision, str):
        normalized = decision.strip().lower()
    else:
        # If decision is not a string, default to "escalate"
        normalized = "escalate"
    if normalized in {"yes", "y", "ok", "approved", "accept"}:
        normalized = "approve"
    elif normalized in {"no", "n", "reject", "deny", "decline"}:
        normalized = "reject"
    elif normalized in {"maybe", "unsure", "idk", "escalate", "needs review"}:
        normalized = "escalate"
    else:
        normalized = "escalate"

    if normalized == "approve":
        final_action = "delete_account"
    elif normalized == "reject":
        final_action = "hold_account"
    else:
        final_action = "hold_account"

    return Command(
        update=state.model_copy(
            update={
                "review_decision": normalized,
                "action": final_action,
            }
        ),
        goto="explain",
    )


@traceable(name="logging_node", tags=["node"])
def logging_node(state: GraphState) -> GraphState:
    # project_id = state.project_id or "panel-app-dev"
    # event_id = state.event_id

    event_id = state.event_id or uuid4().hex

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
        "review_decision": getattr(state, "review_decision", None),
        "timestamp": _utcnow().isoformat(),
    }
    
    if state.explanation_report:
        log_summary["explanation_report"] = state.explanation_report

    logger.info(json.dumps(log_summary, separators=(",", ":")))
    return state.model_copy(
        update={"log_entry": json.dumps(log_summary, ensure_ascii=False, indent=2)}
    )
