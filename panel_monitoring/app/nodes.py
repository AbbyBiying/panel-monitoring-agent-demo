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
from panel_monitoring.app.injection_detector import detect_injection_ml
from panel_monitoring.app.schemas import GraphState, Signals, ModelMeta
from panel_monitoring.app.utils import (
    build_llm_decision_summary_from_signals,
    detect_prompt_injection,
    log_info,
    looks_like_automated,
)
from panel_monitoring.app.retry import firestore_write_with_retry
from panel_monitoring.data.firestore_client import (
    embed_text,
    events_col,
    get_active_prompt_spec,
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


# Fields that the LLM must see first (critical for classification).
# Everything else follows in original order; verbose fields go last.
_PRIORITY_FIELDS = [
    "rule_based_flags",
    "third_party_signals",
    "identity",
    "network_device",
    "recruitment",
    "registration_profile",
]
_DEFERRED_FIELDS = [
    "additional_profile_answers",
]


def _reorder_event(data: dict) -> dict:
    """Reorder event dict so critical fields come first, verbose fields last."""
    ordered: dict = {}
    for key in _PRIORITY_FIELDS:
        if key in data:
            ordered[key] = data[key]
    for key in data:
        if key not in ordered and key not in _DEFERRED_FIELDS:
            ordered[key] = data[key]
    for key in _DEFERRED_FIELDS:
        if key in data:
            ordered[key] = data[key]
    return ordered


def _event_text_from_state(state: GraphState) -> str:
    """Produce a plain-text representation for classification & preview."""
    if isinstance(state.event_text, str):
        return state.event_text
    src = state.event_data or ""
    if isinstance(src, str):
        return src
    try:
        return json.dumps(_reorder_event(src), ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        logger.warning("_event_text_from_state: json.dumps failed, falling back to str()")
        return str(src)


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
    log_info(f"panelist_id={state.panelist_id}")

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
        await firestore_write_with_retry(
            events.document(event_id),
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
        await firestore_write_with_retry(
            evt_ref,
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

            },
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
    t0 = time.perf_counter()

    # Only check human-input fields for garbled text, not the full JSON blob.
    # If no profile fields exist, skip the heuristic and let the LLM decide.
    event_dict = state.event_data if isinstance(state.event_data, dict) else {}
    profile = get_nested_value(event_dict, ["registration_profile"], {}) or {}
    human_input = " ".join(filter(None, [
        profile.get("where_heard_about_us", ""),
        get_nested_value(event_dict, ["identity", "panelist_id"], ""),
    ])).strip()

    prompt_id = None
    prompt_name = None
    injection_flags: list[dict] = []

    if human_input and looks_like_automated(human_input):
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

        # --- Load active PromptSpec from Firestore (required) ---
        prompt_spec = await get_active_prompt_spec("signup_classification")

        if not prompt_spec:
            raise RuntimeError(
                "No live PromptSpec found for role 'signup_classification'. "
                "Push and promote a prompt version before running the agent."
            )

        system_prompt_override = prompt_spec.system_prompt or None
        user_prompt_override = prompt_spec.user_prompt or None
        if prompt_spec.model_name:
            model = prompt_spec.model_name
        prompt_id = prompt_spec.doc_id
        prompt_name = (
            f"{prompt_spec.deployment_role}@v{prompt_spec.version}"
            if prompt_spec.version
            else prompt_spec.deployment_role
        )
        log_info(
            "Loaded PromptSpec from Firestore | id=%s name=%s model=%s",
            prompt_id, prompt_name, model,
        )

        # --- Prompt injection scan on untrusted inputs ---
        # Layer 1: fast regex scan
        event_scan = detect_prompt_injection(text, source="event")
        if event_scan.detected:
            injection_flags.append(event_scan.model_dump())
        for i, doc in enumerate(state.retrieved_docs or [], 1):
            doc_scan = detect_prompt_injection(
                doc.get("text", ""), source=f"retrieved_doc_{i}",
            )
            if doc_scan.detected:
                injection_flags.append(doc_scan.model_dump())

        # Layer 2: DeBERTa ML model (freeform fields only — full JSON
        # triggers false positives on structured data like brackets/colons)
        _freeform_parts: list[str] = []
        if human_input:
            _freeform_parts.append(human_input)
        for answer in (event_dict.get("additional_profile_answers") or []):
            if isinstance(answer, str) and answer.strip():
                _freeform_parts.append(answer.strip())
            elif isinstance(answer, dict):
                ans_text = answer.get("answer") or answer.get("text") or ""
                if isinstance(ans_text, str) and ans_text.strip():
                    _freeform_parts.append(ans_text.strip())
        freeform_text = " ".join(_freeform_parts)

        if freeform_text:
            ml_event = await detect_injection_ml(freeform_text, source="event_freeform")
            if ml_event.detected:
                injection_flags.append({
                    "detected": True,
                    "matched_patterns": [f"ml:{ml_event.label}"],
                    "source": ml_event.source,
                    "ml_score": ml_event.score,
                })
        for i, doc in enumerate(state.retrieved_docs or [], 1):
            ml_doc = await detect_injection_ml(
                doc.get("text", ""), source=f"retrieved_doc_{i}",
            )
            if ml_doc.detected:
                injection_flags.append({
                    "detected": True,
                    "matched_patterns": [f"ml:{ml_doc.label}"],
                    "source": ml_doc.source,
                    "ml_score": ml_doc.score,
                })

        if injection_flags:
            log_info(
                "Prompt injection detected | event_id=%s panelist_id=%s | flags=%s",
                event_id, state.panelist_id, injection_flags,
            )

        log_info(
            f"Evaluating signals with LLM (provider={provider}, model={model}), event_id={event_id} panelist_id={state.panelist_id}"
        )

        try:
            out = await aclassify_event(
                text,
                retrieved_docs=state.retrieved_docs or None,
                system_prompt_override=system_prompt_override,
                user_prompt_override=user_prompt_override,
            )

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

    meta.latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    # ---- Prompt injection override -------------------------------------------
    # Legitimate panelist data should never contain injection patterns.
    # If the LLM was tricked into saying "normal", override to suspicious.
    if injection_flags and signals.normal_signup and not signals.suspicious_signup:
        pattern_names = ", ".join(
            f["matched_patterns"][0]
            for f in injection_flags
            if f.get("matched_patterns") and len(f["matched_patterns"]) > 0
        )
        signals = Signals(
            suspicious_signup=True,
            normal_signup=False,
            confidence=max(float(signals.confidence or 0), 0.85),
            reason=f"Prompt injection detected ({pattern_names}). Original: {signals.reason}",
            panelist_id=signals.panelist_id,
        )

    # ---- Decide high-level classification; action decided later ---------------
    if signals.suspicious_signup and not signals.normal_signup:
        classification: Literal["suspicious", "normal", "uncertain"] = "suspicious"
    elif signals.normal_signup and not signals.suspicious_signup:
        classification = "normal"
    else:
        classification = "uncertain"

    confidence = float(signals.confidence or 0.0)

    action = "no_action"

    result = {
        "signals": signals,
        "classification": classification,
        "confidence": confidence,
        "action": action,
        "model_meta": meta,
        "injection_flags": injection_flags,
    }
    if prompt_id is not None:
        result["prompt_id"] = prompt_id
    if prompt_name is not None:
        result["prompt_name"] = prompt_name
    return result


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
    log_info(f"Action decided | event_id={state.event_id} panelist_id={state.panelist_id} classification={state.classification} confidence={conf:.2f} action={action}")
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
    else:
        final_action = "hold_account"

    # https://docs.langchain.com/oss/python/langgraph/graph-api#command
    return Command(
        update={"review_decision": normalized, "action": final_action},
        goto="explain",
    )


@traceable(name="save_classification_node", tags=["node"])
async def save_classification_node(state: GraphState) -> dict:
    """Persist classification result immediately so it survives interrupt/crash."""
    project_id = state.project_id or "panel-app-dev"
    event_id = state.event_id
    run_id = f"{event_id}_{uuid4().hex[:12]}"
    decision = state.classification or "error"
    confidence = float(state.confidence or 0.0)

    signals_dict = state.signals.model_dump() if state.signals else {}
    meta = state.model_meta
    model_name = meta.model or "gemini-2.5-flash"
    cost_usd = _estimate_cost(model_name, meta.usage or {})

    llm_decision_summary = build_llm_decision_summary_from_signals(
        state.signals,
        confidence_fallback=confidence,
    )

    # --- runs: initial audit record ---
    runs = await runs_col()
    run_data = {
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
        "status": "awaiting_review" if decision == "suspicious" else ("success" if decision != "error" else "error"),
        "llm_decision_summary": llm_decision_summary,
        "started_at": firestore.SERVER_TIMESTAMP,
        "panelist_id": state.panelist_id,
    }
    if state.prompt_id:
        run_data["prompt_id"] = state.prompt_id
    if state.prompt_name:
        run_data["prompt_name"] = state.prompt_name
    await firestore_write_with_retry(runs.document(run_id), run_data)

    # --- events: update with classification (visible in UI immediately) ---
    status = "awaiting_review" if decision == "suspicious" else ("classified" if decision != "error" else "error")
    events = await events_col()
    event_data = {
        "project_id": project_id,
        "status": status,
        "decision": decision,
        "confidence": confidence,
        "signals": signals_dict,
        "last_run_id": run_id,
        "updated_at": firestore.SERVER_TIMESTAMP,
        "panelist_id": state.panelist_id,
    }
    if state.prompt_id:
        event_data["prompt_id"] = state.prompt_id
    if state.prompt_name:
        event_data["prompt_name"] = state.prompt_name
    await firestore_write_with_retry(events.document(event_id), event_data, merge=True)

    log_info(
        "Classification saved | event=%s | panelist=%s | decision=%s | confidence=%.2f | status=%s",
        event_id, state.panelist_id, decision, confidence, status,
    )

    return {"run_id": run_id}


@traceable(name="logging_node", tags=["node"])
async def logging_node(state: GraphState) -> dict:
    """Final save: update existing docs with action, review decision, and explanation."""
    event_id = state.event_id
    run_id = state.run_id
    decision = state.classification or "error"
    confidence = float(state.confidence or 0.0)

    meta = state.model_meta
    model_name = meta.model or "gemini-2.5-flash"
    cost_usd = _estimate_cost(model_name, meta.usage or {})
    explanation = state.explanation_report or []

    llm_decision_summary = build_llm_decision_summary_from_signals(
        state.signals,
        confidence_fallback=confidence,
    )

    # --- runs: update with final action & review outcome ---
    runs = await runs_col()
    await firestore_write_with_retry(
        runs.document(run_id),
        {
            "action": state.action or "no_action",
            "review_decision": getattr(state, "review_decision", None),
            "explanation_report": explanation,
            "status": "success" if decision != "error" else "error",
            "completed_at": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )

    # --- events: update with final action ---
    events = await events_col()
    await firestore_write_with_retry(
        events.document(event_id),
        {
            "status": "classified" if decision != "error" else "error",
            "action": state.action or "no_action",
            "updated_at": firestore.SERVER_TIMESTAMP,
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

    log_info(f"Run complete | event_id={event_id} panelist_id={state.panelist_id} decision={decision} confidence={confidence:.2f} action={state.action} model={model_name}")
    return {"log_entry": json.dumps(log_summary, ensure_ascii=False, indent=2)}
