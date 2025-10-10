# panel_monitoring/app/graph.py

from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from panel_monitoring.app.schemas import GraphState
from panel_monitoring.app.nodes import (
    user_event_node,
    action_decision_node,
    explanation_node,
    logging_node,
    make_signal_eval_node,
)
from panel_monitoring.app.clients.llms.provider_base import ClassifierProvider


def build_graph(provider: ClassifierProvider):
    """
    Build the LangGraph workflow with the supplied classifier provider.

    Expected state keys during execution:
      - set by `user_event_node`:   project_id, event_id, event_text/event_data
      - set by `make_signal_eval_node(provider)`: signals, classification, confidence, model_meta
      - set by `action_decision_node`: action (optional)
      - set by `explanation_node`: explanation_report (optional)
      - set by `logging_node`: persists run + event summary; may add log_entry
    """
    wf = StateGraph(GraphState)

    # Nodes
    wf.add_node("event_input", user_event_node)
    wf.add_node("classify_signals", make_signal_eval_node(provider))
    wf.add_node("decide_action", action_decision_node)
    wf.add_node("explain", explanation_node)
    wf.add_node("log_result", logging_node)

    # Routing after classification
    def route_to_action(state: GraphState) -> str:
        cls = state.get("classification")  # "suspicious" | "normal" | "error"
        if cls == "suspicious":
            return "decide_action"
        # For "normal" and any unexpected/None classification, explain and proceed to logging.
        return "explain"

    # Edges
    wf.add_edge(START, "event_input")
    wf.add_edge("event_input", "classify_signals")
    wf.add_conditional_edges(
        "classify_signals",
        route_to_action,
        {"decide_action": "decide_action", "explain": "explain"},
    )
    wf.add_edge("decide_action", "explain")
    wf.add_edge("explain", "log_result")
    wf.add_edge("log_result", END)

    return wf.compile()
