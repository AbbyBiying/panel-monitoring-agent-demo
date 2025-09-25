# https://colab.research.google.com/drive/10Elz5uCWVV3xKcFSJW0QWFWSDbopAqHT


# In[2]: Define the Graph State and Nodes
# This cell contains the core logic of our agent:
# - The GraphState, which acts as the shared memory.
# - The four nodes that process the event data in a linear sequence.

from typing import TypedDict, Callable
from langgraph.graph import StateGraph, START, END
import random
import json

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        event_data (str): The raw user event data (e.g., a new signup).
        signals (dict): A dictionary of signals classified from the event.
        action (str): The final action decided upon (e.g., 'remove_account').
        log_entry (str): A summary of the event for logging and reporting.
    """
    event_data: str
    signals: dict
    action: str
    log_entry: str

def user_event_node(state: GraphState) -> GraphState:
    """Initial node to receive the user event and format it."""
    print("--- 1. User Event Node: Received new event ---")
    return {"event_data": state.get("event_data", ""), "signals": {}, "action": "", "log_entry": ""}

def signal_evaluation_node(state: GraphState) -> GraphState:
    """Simulates an LLM call to evaluate the event and classify signals."""
    print("--- 2. Signal Evaluation Node: Analyzing event for signals ---")
    event = state.get("event_data", "")
    signals = {}
    if "suspicious" in event.lower() or ("director" in event.lower() and "22" in event.lower()):
        signals["suspicious_signup"] = True
    else:
        signals["normal_signup"] = True
    return {"signals": signals}

def action_decision_node(state: GraphState) -> GraphState:
    """Decides on a course of action based on the detected signals."""
    print("--- 3. Action Decision Node: Deciding on account action ---")
    signals = state.get("signals", {})
    action = "no_action"
    if signals.get("suspicious_signup"):
        if random.random() > 0.5:
            action = "remove_account"
        else:
            action = "hold_account"
    else:
        action = "no_action"
    return {"action": action}

def logging_node(state: GraphState) -> GraphState:
    """Final node to log the event and the outcome."""
    print("--- 4. Logging Node: Logging final event outcome ---")
    log_summary = {
        "event": state.get("event_data", "N/A"),
        "signals_detected": state.get("signals", {}),
        "final_action": state.get("action", "N/A")
    }
    log_entry = json.dumps(log_summary, indent=2)
    print("\n--- Final Log Entry ---")
    print(log_entry)
    return {"log_entry": log_entry}

# In[3]: Assemble and Compile the Graph
# This cell builds the graph and makes it ready to be executed.

# Build the graph
workflow = StateGraph(GraphState)
workflow.add_node("user_event_node", user_event_node)
workflow.add_node("signal_evaluation_node", signal_evaluation_node)
workflow.add_node("action_decision_node", action_decision_node)
workflow.add_node("logging_node", logging_node)

# Add the linear edges
workflow.add_edge(START, "user_event_node")
workflow.add_edge("user_event_node", "signal_evaluation_node")
workflow.add_edge("signal_evaluation_node", "action_decision_node")
workflow.add_edge("action_decision_node", "logging_node")
workflow.add_edge("logging_node", END)

# Compile the graph into a runnable object
app = workflow.compile()
print("Graph compiled successfully. Ready for testing!")

# In[4]: Test the Graph with Different Events
# This is where you can run your agent and observe the output.
# You can change the 'suspicious_event' or 'normal_event' strings to test new inputs.

# Test scenario 1: A suspicious user event
print("==========================================================")
print("TESTING SCENARIO 1: SUSPICIOUS SIGNUP")
print("==========================================================")
suspicious_event = "NEW SIGNUP: I am a 22 year old Director making $200,000 per year."
final_state_1 = app.invoke({"event_data": suspicious_event})

# Test scenario 2: A normal user event
print("\n\n==========================================================")
print("TESTING SCENARIO 2: NORMAL SIGNUP")
print("==========================================================")
normal_event = "NEW SIGNUP: I am a 35 year old teacher making $50,000 per year."
final_state_2 = app.invoke({"event_data": normal_event})