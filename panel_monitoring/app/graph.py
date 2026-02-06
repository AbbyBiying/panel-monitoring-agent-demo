# panel_monitoring/app/graph.py
from __future__ import annotations

import logging

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from panel_monitoring.app.schemas import GraphState
from panel_monitoring.app.nodes import (
    perform_effects_node,
    user_event_node,
    retrieval_node,
    signal_evaluation_node,
    action_decision_node,
    human_approval_node,
    explanation_node,
    logging_node,
)
from panel_monitoring.app.clients.llms import init_llm_client

logger = logging.getLogger(__name__)

# Pre-initialize LLM client at module import time (before async event loop)
# This ensures all blocking I/O (credentials, .env files) happens synchronously
init_llm_client()


def build_graph():
    """
    Construct the LangGraph workflow for the panel monitoring agent.
    Compiles with a checkpointer so human-in-the-loop interrupts can pause/resume.
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

    def route_from_classify(state: GraphState) -> str:
        # Determine next node based on classification result from signal evaluation

        c = (state.classification or "").lower()
        # Pending: not classified yet so run explanation or classification step
        if c == "pending":
            logger.info("classification_pending")
            return "explain"

        if c == "normal":
            return "explain"

        if c in ("suspicious", "error"):
            if (state.confidence is not None) and (state.confidence < 0.30):
                logger.warning("low_confidence:%.2f", state.confidence)
            return "decide_action"

        # Unexpected classification value – still go to action but log it
        logger.warning("unknown_classification:%s", state.classification)
        return "decide_action"

    def route_after_decide(state: GraphState) -> str:
        a = (state.action or "").lower()
        if a == "request_human_review":
            return "human_approval"
        # If a review decision was already made, proceed to apply effects
        if state.review_decision is not None:
            return "perform_effects"
        # Default to explain; effects can be no-ops if action is empty
        return "explain"

    # --- graph construction ---
    graph.add_node("event_input", user_event_node)
    graph.add_node("retrieve_context", retrieval_node)
    graph.add_node("classify_signals", signal_evaluation_node)
    graph.add_node("decide_action", action_decision_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("explain", explanation_node)
    graph.add_node("perform_effects", perform_effects_node)
    graph.add_node("log_result", logging_node)

    graph.add_edge(START, "event_input")
    graph.add_edge("event_input", "retrieve_context")
    graph.add_edge("retrieve_context", "classify_signals")

    graph.add_conditional_edges(
        "classify_signals",
        route_from_classify,
        {"decide_action": "decide_action", "explain": "explain"},
    )

    graph.add_conditional_edges(
        "decide_action",
        route_after_decide,
        {
            "human_approval": "human_approval",
            "perform_effects": "perform_effects",
            "explain": "explain",
        },
    )

    graph.add_edge("human_approval", "explain")
    graph.add_edge("explain", "perform_effects")  # <-- run effects after explanation
    graph.add_edge("perform_effects", "log_result")
    graph.add_edge("log_result", END)

    # checkpointer required for interrupts
    saver = MemorySaver()
    compiled = graph.compile(checkpointer=saver)

    return compiled
