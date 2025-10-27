# panel_monitoring/app/graph.py
from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from panel_monitoring.app.schemas import GraphState
from panel_monitoring.app.nodes import (
    perform_effects_node,
    user_event_node,
    signal_evaluation_node,
    action_decision_node,
    human_approval_node,
    explanation_node,
    logging_node,
)


def build_graph():
    """
    Construct the LangGraph workflow for the panel monitoring agent.
    Compiles with a checkpointer so human-in-the-loop interrupts can pause/resume.
    """
    graph = StateGraph(GraphState)

    # --- routing helpers ---
    def route_from_classify(state: GraphState) -> str:
        # If the classifier said "normal", skip to explain
        if state.classification == "normal":
            return "explain"
        # For anything else (including "suspicious" or "error"), let decision logic set `action`
        return "decide_action"

    def route_after_decide(state: GraphState) -> str:
        # Only gate when action requests review
        return (
            "human_approval" if (state.action == "request_human_review") else "explain"
        )

    # --- nodes ---
    graph.add_node("event_input", user_event_node)
    graph.add_node("classify_signals", signal_evaluation_node)
    graph.add_node("decide_action", action_decision_node)
    graph.add_node("human_approval", human_approval_node)  # pauses if review needed
    graph.add_node("explain", explanation_node)
    graph.add_node("perform_effects", perform_effects_node)  # <-- now actually used
    graph.add_node("log_result", logging_node)

    # --- edges ---
    graph.add_edge(START, "event_input")
    graph.add_edge("event_input", "classify_signals")

    graph.add_conditional_edges(
        "classify_signals",
        route_from_classify,
        {"decide_action": "decide_action", "explain": "explain"},
    )

    graph.add_conditional_edges(
        "decide_action",
        route_after_decide,
        {"human_approval": "human_approval", "explain": "explain"},
    )

    graph.add_edge("human_approval", "explain")
    graph.add_edge("explain", "perform_effects")  # <-- run effects after explanation
    graph.add_edge("perform_effects", "log_result")
    graph.add_edge("log_result", END)

    # checkpointer required for interrupts
    saver = MemorySaver()
    compiled = graph.compile(checkpointer=saver)

    return compiled
