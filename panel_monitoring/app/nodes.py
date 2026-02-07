# panel_monitoring/app/nodes.py
from __future__ import annotations

import json
import time
from datetime import datetime, UTC
import os
from typing import Literal, Tuple
from uuid import uuid4

from google.cloud import firestore
from langsmith import traceable
from langgraph.types import Command, interrupt

import logging

from panel_monitoring.app.clients.llms import aclassify_event
from panel_monitoring.app.rules import apply_occupation_rules
from panel_monitoring.app.schemas import GraphState, Signals, ModelMeta
from panel_monitoring.app.utils import (
    build_llm_decision_summary_from_signals,
    log_info,
    looks_like_automated,
)
from panel_monitoring.data.firestore_client import (
    embed_text,
    events_col,
    get_similar_patterns,
    runs_col,
)

# --------------------------------------------------------------------
# Confidence Thresholds for Automated Decision Making
# --------------------------------------------------------------------

# If confidence is ≥ this value → auto-approve the model’s action (no human needed)
CONFIDENCE_AUTO_APPROVE_THRESHOLD = 0.90

# If confidence is ≥ this value → require Human-In-The-Loop (HITL) review
CONFIDENCE_REVIEW_THRESHOLD = 0.70

# If confidence is between 0.30 and 0.69 → model is uncertain, hold/no-action
CONFIDENCE_UNCERTAIN_THRESHOLD = 0.30

HEURISTIC_CONFIDENCE = 0.70

# Pricing per 1M tokens (USD) — update when Google changes pricing
# https://cloud.google.com/vertex-ai/generative-ai/pricing
_COST_PER_1M = {
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-pro":   {"input": 1.25, "output": 10.00},
}


logger = logging.getLogger(__name__)


# --- helpers ---------------------------------------------------------------

def _estimate_cost(model: str, usage: dict) -> float | None:
    """Estimate USD cost from token counts. Returns None if data is missing."""
    rates = _COST_PER_1M.get(model)
    if not rates or not usage:
        return None
    prompt_tokens = usage.get("prompt_token_count") or usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("candidates_token_count") or usage.get("completion_tokens") or 0
    if not (prompt_tokens or completion_tokens):
        return None
    return round(
        prompt_tokens / 1_000_000 * rates["input"]
        + completion_tokens / 1_000_000 * rates["output"],
        6,
    )


def get_nested_value(data, keys, default=None):
    """Safe navigation for deeply nested dictionaries."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        else:
            return default
    return data

def extract_panel_data(raw_data):
    """
    Normalizes messy input. 
    If input is a list, it only processes the first element.
    """
    # 1. Convert string to object
    if isinstance(raw_data, str):
        try:
            raw_data = json.loads(raw_data)
        except (json.JSONDecodeError, ValueError, TypeError):
            return {}

    # 2. Handle List: Take the first element only
    if isinstance(raw_data, list):
        raw_data = raw_data[0] if len(raw_data) > 0 else {}
    
    # 3. Final safety check: must be a dict to continue
    if not isinstance(raw_data, dict):
        return {}

    # Path A: Flat (New Format)
    identity = raw_data.get("identity")
    
    # Path B: Nested (Previous Format)
    if not identity:
        identity = get_nested_value(raw_data, ["input", "identity"])

    panelist_id = (identity or {}).get("panelist_id") or raw_data.get("id")
    email_domain = (identity or {}).get("primary_email_domain")
    
    return {
        "panelist_id": panelist_id,
        "identity": identity or {},
        "raw": raw_data,
        "email_domain": email_domain
    }

def _utcnow() -> datetime:
    return datetime.now(UTC)


def _heuristic_fallback(text: str) -> Tuple[Signals, ModelMeta]:
    t = (text or "").lower()
    flagged = any(k in t for k in ("suspicious", "fraud", "bot", "abuse", "disposable"))
    return (
        Signals(
            suspicious_signup=flagged,
            normal_signup=not flagged,
            confidence=HEURISTIC_CONFIDENCE,
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
async def perform_effects_node(state: GraphState) -> dict:
    try:
        if state.action == "delete_account":
            logger.info("Deleting account for event_id=%s panelist_id=%s", state.event_id, state.panelist_id)
            # Wrap string in a list []
            return {"explanation_report": ["[INFO] effect:delete_account executed"]}

    except Exception as e:
        logger.warning("Effect execution failed for event_id=%s panelist_id=%s: %s", state.event_id, state.panelist_id, e)
        line = f"[ERROR] effect_failed:{type(e).__name__} {e}"
        # Wrap string in a list []
        return {
            "explanation_report": [line],
            "action": "hold_account",
        }
    return {}


@traceable(name="user_event_node", tags=["node"])
async def user_event_node(state: GraphState) -> dict:
    project_id = state.project_id or os.getenv("PANEL_PROJECT_ID", "panel-app-dev")
    event_text = _event_text_from_state(state)

    # 1. Normalize the messy input into a clean dict
    clean_data = extract_panel_data(state.event_data)
    
    panelist_id = clean_data.get("panelist_id")   

    seeded_signals = Signals(
        suspicious_signup=False,
        normal_signup=True,
        confidence=0.0,
        reason="unclassified",
        panelist_id=panelist_id
    )

    events = await events_col()
    if state.event_id:
        event_id = state.event_id
        # Only update mutable fields on existing doc
        await events.document(event_id).set(
            {
                "project_id": project_id,
                "updated_at": firestore.SERVER_TIMESTAMP,
                "status": "pending",
                "payload": {"preview": (event_text or "")[:200]},
                "panelist_id": panelist_id
            },
            merge=True,
        )
    else:
        # Create new doc with immutable creation fields
        evt_ref = events.document()
        await evt_ref.set(
            {
                "project_id": project_id,
                "type": getattr(state, "event_type", "signup"),
                "source": getattr(state, "event_source", "web"),
                "received_at": firestore.SERVER_TIMESTAMP,
                "updated_at": firestore.SERVER_TIMESTAMP,
                "event_at": _utcnow(),
                "status": "pending",
                "payload": {"preview": (event_text or "")[:200]},
                "panelist_id": panelist_id

            }
        )
        event_id = evt_ref.id

    return {
        "project_id": project_id,
        "event_id": event_id,
        "event_text": event_text,
        "signals": seeded_signals,
        "classification": "pending",
        "action": None,
        "log_entry": "",
        # RESET the report so old decisions don't stick around
        "explanation_report": [],
        "panelist_id": panelist_id
    }


@traceable(name="retrieval_node", tags=["node"])
async def retrieval_node(state: GraphState) -> dict:
    """Embed event text and retrieve similar fraud patterns from Firestore."""
    text = _event_text_from_state(state)
    if not text:
        return {}

    try:
        query_vector = await embed_text(text)
        docs = await get_similar_patterns(query_vector, limit=5)
        log_info(f"Retrieved {len(docs)} similar patterns for event_id={state.event_id} panelist_id={state.panelist_id}")
        return {"retrieved_docs": docs}
    except Exception as e:
        logger.warning("RAG retrieval failed for event_id=%s panelist_id=%s: %s", state.event_id, state.panelist_id, e)
        return {}


@traceable(name="signal_evaluation_node", tags=["node"])
async def signal_evaluation_node(state: GraphState) -> dict:
    text = _event_text_from_state(state)
    event_id = state.event_id or "unknown"

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
        model = os.getenv("VERTEX_MODEL", "gemini-2.5-flash")

        log_info(
            f"Evaluating signals with LLM (provider={provider}, model={model}), event_id={event_id} panelist_id={state.panelist_id}"
        )

        try:
            t0 = time.perf_counter()
            out = await aclassify_event(text, retrieved_docs=state.retrieved_docs or None)
            latency_ms = int((time.perf_counter() - t0) * 1000)

            if isinstance(out, tuple) and len(out) == 2:
                raw_sig, raw_meta = out
            else:
                raw_sig, raw_meta = (out or {}), {}

            if not isinstance(raw_sig, dict):
                raw_sig = {}

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
                raw_sig["reason"] = (
                    raw_sig.get("message") or raw_meta.get("reason") or ""
                )

            signals = Signals.model_validate(raw_sig)
            meta = ModelMeta.model_validate(raw_meta or {})
            meta.latency_ms = latency_ms

            if not getattr(meta, "provider", None):
                meta.provider = provider
            if not getattr(meta, "model", None):
                meta.model = model

            log_info(
                "LLM classification | event=%s | panelist=%s | suspicious=%s | confidence=%.2f | provider=%s | model=%s | reason=%s",
                event_id,
                state.panelist_id,
                signals.suspicious_signup,
                float(signals.confidence or 0),
                meta.provider,
                meta.model,
                signals.reason,
            )

        except Exception as e:
            logger.warning("LLM classification failed for event_id=%s panelist_id=%s; using heuristic fallback.", event_id, state.panelist_id)
            log_info(f"LLM classification error for event_id={event_id} panelist_id={state.panelist_id}: {e}")
            signals, meta = _heuristic_fallback(text)
            signals.reason = f"{signals.reason} ({type(e).__name__})"
    # ---- Rule-based post-processing / guardrails -----------------------------
    raw_signals_dict = signals.model_dump()
    raw_signals_dict = apply_occupation_rules(
        raw_signals_dict,
        event_text=text,
        event_id=event_id,
    )
    signals = Signals.model_validate(raw_signals_dict)

    # ---- Decide high-level classification; action decided later ---------------
    if signals.suspicious_signup and not signals.normal_signup:
        classification: Literal["suspicious", "normal", "uncertain"] = "suspicious"
    elif signals.normal_signup and not signals.suspicious_signup:
        classification = "normal"
    else:
        classification = "uncertain"

    confidence = float(signals.confidence or 0.0)

    action = "no_action"

    return {
        "signals": signals,
        "classification": classification,
        "confidence": confidence,
        "action": action,
        "model_meta": meta,
    }


@traceable(name="action_decision_node", tags=["node"])
async def action_decision_node(state: GraphState) -> dict:
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
    return {"action": action}


@traceable(name="explanation_node", tags=["node"])
async def explanation_node(state: GraphState) -> dict:
    cls = state.classification
    sig = state.signals
    action = state.action or "no_action"

    # safe confidence string
    conf_val = (
        sig.confidence if (sig and sig.confidence is not None) else state.confidence
    )
    conf_txt = f"{float(conf_val):.2f}" if isinstance(conf_val, (int, float)) else "N/A"

    if cls == "suspicious":
        reason = sig.reason if (sig and sig.reason) else "n/a"
        ex = f"SUSPICIOUS (Conf {conf_txt}). Reason: {reason}. Action: {action}."
    elif cls == "normal":
        ex = f"NORMAL (Conf {conf_txt}). Action: {action}."
    else:
        ex = "Classification error."

    if action == "request_human_review" and getattr(state, "review_url", None):
        ex = f"{ex} Pending human review → {state.review_url}"

    if ex in state.explanation_report:
        return {}

    return {"explanation_report": [ex]}


@traceable(name="human_approval_node", tags=["node"])
async def human_approval_node(
    state: GraphState,
) -> dict | Command[Literal["explain"]]:
    """
    If we requested human review, PAUSE here and ask an admin.
    When resumed, convert the review decision into the final action.
    """
    if state.action != "request_human_review":
        # Not a gated path; continue downstream
        return {}

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

    # https://docs.langchain.com/oss/python/langgraph/graph-api#command
    return Command(
        update={"review_decision": normalized, "action": final_action},
        goto="explain",
    )


@traceable(name="logging_node", tags=["node"])
async def logging_node(state: GraphState) -> dict:
    project_id = state.project_id or "panel-app-dev"
    event_id = state.event_id

    # #6: Prefix run_id with event_id for easier correlation
    run_id = f"{event_id}_{uuid4().hex[:12]}"
    decision = state.classification or "error"
    confidence = float(state.confidence or 0.0)

    # Convert models to dicts for Firestore I/O
    signals_dict = state.signals.model_dump() if state.signals else {}
    meta = state.model_meta
    model_name = meta.model or "gemini-2.5-flash"

    llm_decision_summary = build_llm_decision_summary_from_signals(
        state.signals,
        confidence_fallback=confidence,
    )
    explanation = state.explanation_report or []

    # #4: Calculate cost from token usage instead of storing null
    cost_usd = _estimate_cost(model_name, meta.usage or {})

    # --- runs: full immutable audit trail (one per classification attempt) ---
    runs = await runs_col()
    await runs.document(run_id).set(
        {
            "run_id": run_id,
            "project_id": project_id,
            "event_id": event_id,
            "provider": meta.provider or "vertexai",
            "model": model_name,
            "decision": decision,
            "confidence": confidence,
            "signals": signals_dict,
            "latency_ms": meta.latency_ms,
            "cost_usd": cost_usd,
            "usage": meta.usage or {},
            "status": "success" if decision != "error" else "error",
            "llm_decision_summary": llm_decision_summary,
            "action": state.action or "no_action",
            "review_decision": getattr(state, "review_decision", None),
            "explanation_report": explanation,
            "started_at": firestore.SERVER_TIMESTAMP,
            "panelist_id": state.panelist_id,
        }
    )
    # --- events: latest snapshot for UI list views & filtering ---
    # #2: Keep only what the UI needs for list/filter. Full detail via last_run_id → runs.
    # #3: Include signals so reviewers see suspicious/normal breakdown without joining.
    events = await events_col()
    await events.document(event_id).set(
        {
            "project_id": project_id,
            "status": "classified" if decision != "error" else "error",
            "decision": decision,
            "confidence": confidence,
            "action": state.action or "no_action",
            "signals": signals_dict,
            "last_run_id": run_id,
            "updated_at": firestore.SERVER_TIMESTAMP,
            "panelist_id": state.panelist_id,
        },
        merge=True,
    )

    # Build human-friendly log summary
    preview = _event_text_from_state(state)[:50]

    log_summary = {
        "event_id": event_id,
        "panelist_id": state.panelist_id,
        "run_id": run_id,
        "classification": decision,
        "confidence": confidence,
        "final_action": state.action
        or ("no_action" if decision == "normal" else "N/A"),
        "provider": meta.provider or "NONE",
        "model": model_name,
        "latency_ms": meta.latency_ms,
        "cost_usd": cost_usd,
        "event_preview": f"{preview}...",
        "review_decision": getattr(state, "review_decision", None),
        "timestamp": _utcnow().isoformat(),
        "llm_decision_summary": llm_decision_summary,
    }

    if state.explanation_report:
        log_summary["explanation_report"] = state.explanation_report

    logger.info(json.dumps(log_summary, separators=(",", ":")))
    return {"log_entry": json.dumps(log_summary, ensure_ascii=False, indent=2)}
