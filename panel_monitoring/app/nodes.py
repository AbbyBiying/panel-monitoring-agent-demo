# app/nodes.py
import random
import json
from datetime import datetime, timezone
from typing import Dict
from langsmith import traceable
from panel_monitoring.app.schemas import GraphState
from panel_monitoring.app.utils import looks_like_automated
from panel_monitoring.app.clients.llms.provider_base import ClassifierProvider

@traceable(tags=["node"])
def user_event_node(state: GraphState) -> GraphState:
    return {
        "event_data": state.get("event_data", ""),
        "signals": {},
        "action": "",
        "log_entry": "",
        "classification": "error",
    }


def _heuristic_fallback(event: str) -> Dict:
    ev = (event or "").lower()
    heur = any(k in ev for k in ("suspicious", "fraud", "bot", "abuse", "disposable"))
    return {
        "suspicious_signup": heur,
        "normal_signup": not heur,
        "confidence": 0.5,
        "reason": "LLM failed, used heuristic fallback.",
    }


def make_signal_eval_node(provider: ClassifierProvider):
    @traceable(tags=["node"])
    def signal_evaluation_node(state: GraphState) -> GraphState:
        event = state.get("event_data", "")
        if looks_like_automated(event):
            signals = {
                "suspicious_signup": True,
                "normal_signup": False,
                "confidence": 0.9,
                "reason": "Automated/garbled input.",
            }
        else:
            try:
                signals = provider(event)
            except Exception:
                signals = _heuristic_fallback(event)
        classification = "suspicious" if signals.get("suspicious_signup") else "normal"
        action = "no_action" if classification == "normal" else ""

        return {"signals": signals, "classification": classification, "action": action}

    return signal_evaluation_node


@traceable(tags=["node"])
def action_decision_node(state: GraphState) -> GraphState:
    if state.get("classification") == "suspicious":
        if random.random() > 0.5:
            action = "remove_account"
        else:
            action = "hold_account"
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
        ex = f"The event was flagged as **SUSPICIOUS** (Confidence: {conf_txt}).\nPrimary concern: **{s.get('reason', 'n/a')}**\nFinal action: **{a}**."
    elif c == "normal":
        ex = f"The event was classified as **NORMAL** (Confidence: {conf_txt}). No sufficient evidence of fraud; action: **{a}**."
    else:
        ex = "Classification error occurred; see logs."
    return {"explanation_report": ex}


@traceable(tags=["node"])
def logging_node(state: GraphState) -> GraphState:
    action = state.get("action") or (
        "no_action" if state.get("classification") == "normal" else "N/A"
    )
    log_summary = {
        "event_summary": (state.get("event_data") or "N/A")[:50] + "...",
        "classification": state.get("classification"),
        "signals_detected": state.get("signals", {}),
        "final_action": action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return {"log_entry": json.dumps(log_summary, indent=2)}
