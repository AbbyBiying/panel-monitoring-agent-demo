# panel_monitoring/app/graph.py
from __future__ import annotations
from typing import Dict, Any, Callable
from langgraph.graph import StateGraph, START, END
from panel_monitoring.app.nodes import (
    user_event_node,
    signal_evaluation_node,
    action_decision_node,
    explanation_node,
    logging_node,
    set_classifier_provider,
)
from panel_monitoring.app.clients.llms.provider_base import ClassifierProvider

def _wrap_merge(fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def _wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
        patch = fn(state) or {}
        new_state = dict(state)
        new_state.update(patch)
        return new_state
    return _wrapped

def build_graph(provider: ClassifierProvider):
    set_classifier_provider(provider)

    wf = StateGraph(dict)
    wf.add_node("event_input", _wrap_merge(user_event_node))
    wf.add_node("classify_signals", _wrap_merge(signal_evaluation_node))
    wf.add_node("decide_action", _wrap_merge(action_decision_node))
    wf.add_node("explain", _wrap_merge(explanation_node))
    wf.add_node("log_result", _wrap_merge(logging_node))

    def route_to_action(state: Dict[str, Any]) -> str:
        return "decide_action" if state.get("classification") == "suspicious" else "explain"

    wf.add_edge(START, "event_input")
    wf.add_edge("event_input", "classify_signals")
    wf.add_conditional_edges("classify_signals", route_to_action,
                             {"decide_action": "decide_action", "explain": "explain"})
    wf.add_edge("decide_action", "explain")
    wf.add_edge("explain", "log_result")
    wf.add_edge("log_result", END)
    return wf.compile()
