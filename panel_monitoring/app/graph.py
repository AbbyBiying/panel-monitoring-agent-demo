# app/graph.py
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
    wf = StateGraph(GraphState)
    wf.add_node("event_input", user_event_node)
    wf.add_node("classify_signals", make_signal_eval_node(provider))
    wf.add_node("decide_action", action_decision_node)
    wf.add_node("explain", explanation_node)
    wf.add_node("log_result", logging_node)

    def route_to_action(state: GraphState) -> str:
        return (
            "decide_action"
            if state.get("classification") == "suspicious"
            else "explain"
        )

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
