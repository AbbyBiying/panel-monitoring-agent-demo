# panel_monitoring/app/graph.py
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from panel_monitoring.app.schemas import GraphState
from panel_monitoring.app.nodes import (
    user_event_node,
    signal_evaluation_node,
    action_decision_node,
    explanation_node,
    logging_node,
)


def build_graph():
    """
    Construct the LangGraph workflow for the panel monitoring agent.
    This function builds and connects all workflow nodes. The classifier/provider
    is supplied at runtime via LangGraph *runtime context* (passed on `invoke`),
    not injected here.

    It uses a Pydantic `GraphState` schema so that LangGraph can automatically
    merge partial state updates (dict patches) returned by nodes as the graph executes.
    https://docs.langchain.com/oss/python/langgraph/graph-api#state
    All Nodes will emit updates to the State which are then applied using the specified reducer function.
    https://docs.langchain.com/oss/python/langgraph/graph-api#reducers
    Reducers are key to understanding how updates from nodes are applied to the State.
    Each key in the State has its own independent reducer function.
    If no reducer function is explicitly specified then
    it is assumed that all updates to that key should override it.
    https://docs.langchain.com/oss/python/langgraph/graph-api#default-reducer
    """

    graph = StateGraph(GraphState)

    graph.add_node("event_input", user_event_node)
    graph.add_node("classify_signals", signal_evaluation_node)
    graph.add_node("decide_action", action_decision_node)
    graph.add_node("explain", explanation_node)
    graph.add_node("log_result", logging_node)

    def route_to_action(state: GraphState) -> str:
        return "decide_action" if state.classification == "suspicious" else "explain"

    graph.add_edge(START, "event_input")
    graph.add_edge("event_input", "classify_signals")
    graph.add_conditional_edges(
        "classify_signals",
        route_to_action,
        {"decide_action": "decide_action", "explain": "explain"},
    )
    graph.add_edge("decide_action", "explain")
    graph.add_edge("explain", "log_result")
    graph.add_edge("log_result", END)

    return graph.compile()
