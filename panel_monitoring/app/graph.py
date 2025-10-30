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


    def add_diag_line(state, level, msg):
        line = f"[{level.upper()}] {msg}"
        state.explanation_report = (state.explanation_report or "") + ("\n" if state.explanation_report else "") + line

    
    def route_from_classify(state: GraphState) -> str:
        c = (state.classification or "").lower()
        if c == "normal":
            return "explain"
        if c in ("suspicious", "error"):
            if (state.confidence is not None) and (state.confidence < 0.30):
                add_diag_line(state, "warning", f"low_confidence:{state.confidence:.2f}")
            return "decide_action"

        add_diag_line(state, "warning", f"unknown_classification:{state.classification}")
        return "decide_action"

    
    def route_after_decide(state: GraphState) -> str:
        a = (state.action or "").lower()
        if a == "request_human_review":
            return "human_approval"
        # Default to explain; effects can be no-ops if action is empty
        return "explain"
    
    # --- graph construction ---
    graph.add_node("event_input", user_event_node)
    graph.add_node("classify_signals", signal_evaluation_node)
    graph.add_node("decide_action", action_decision_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("explain", explanation_node)
    graph.add_node("perform_effects", perform_effects_node)
    graph.add_node("log_result", logging_node)

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
