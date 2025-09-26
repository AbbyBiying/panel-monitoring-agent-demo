# panel_agent.py
# LangGraph Agent with Conditional Routing and Robust OpenAI Classification

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from openai import OpenAI, APIError
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langsmith import traceable, Client
from langsmith.run_helpers import trace
import os, json, random
import logging
from datetime import datetime
from uuid import uuid4
from dotenv import load_dotenv, find_dotenv
import time # Added for logging timestamp

# --- SETUP AND CONFIGURATION ---

# Load environment variables (OPENAI_API_KEY, LANGSMITH_API_KEY, etc.)
load_dotenv(find_dotenv(), override=True) # Use override=True for consistent notebook behavior

# Configure LangSmith Project (Ensures tracing works cleanly)
os.environ.pop("LANGSMITH_SESSION", None)
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
PROJECT = os.getenv("LANGSMITH_PROJECT") or f"Panel Monitoring Agent ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
os.environ["LANGSMITH_PROJECT"] = PROJECT

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM Client Setup
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# ChatOpenAI uses env variables like OPENAI_API_KEY automatically
llm = ChatOpenAI(model=MODEL, temperature=0) 

# --- PROMPT AND SCHEMA DEFINITIONS ---

# Pydantic Schema for Structured Output
class Signals(BaseModel):
    suspicious_signup: bool = Field(..., description="True if the event is suspicious")
    normal_signup: bool = Field(..., description="True if the event is normal")
    confidence: float = Field(..., description="A confidence score between 0.0 and 1.0.")
    reason: str = Field(..., description="A brief reason for the classification.")
    
# LLM instance configured for structured output
structured = llm.with_structured_output(Signals)

PROMPT_CLASSIFY_SYSTEM = (
    "You are a dedicated fraud and abuse detection expert for a consumer survey panel. "
    "Your only function is to classify user signup events. ALWAYS return only the JSON schema provided."
)

PROMPT_CLASSIFY_USER = """
Analyze the following panelist event data for signals of fraud or abuse.

1.  Return the classification in the provided JSON schema. Ensure exactly one of "suspicious_signup" or "normal_signup" is true.
2.  "suspicious_signup" must be TRUE if the event shows characteristics of bots, identity theft, or policy abuse (e.g., using a disposable email, rapid-fire activity, or using known fraudulent keywords).
3.  "normal_signup" must be TRUE ONLY if the event appears to be from a legitimate, unique user.

Event Data: {event}
"""

# --- STATE and EXCEPTIONS ---

class LLMClassificationError(Exception):
    """Raised when the LLM response is invalid, ambiguous, or fails."""
    pass

class GraphState(TypedDict):
    event_data: str
    signals: dict 
    classification: Literal["suspicious", "normal", "error"]
    action: str
    log_entry: str
    explanation_report: str

# --- HELPER FUNCTIONS ---

@traceable(project_name=PROJECT, tags=["node"])
def explanation_node(state: GraphState) -> GraphState:
    print("--- 4. Explanation Node ---")
    
    classification = state.get("classification")
    signals = state.get("signals", {})
    action = state.get("action")
    
    if classification == "suspicious":
        reason = signals.get('reason', 'No specific reason provided by the classifier.')
        confidence = signals.get('confidence', 'N/A')
        
        explanation = (
            f"The event was flagged as **{classification.upper()}** (Confidence: {confidence:.2f}).\n"
            f"The primary concern is: **{reason}**\n"
            f"The final action taken was: **{action}**."
        )
    else:
        explanation = (
            f"The event was classified as **{classification.upper()}**. "
            f"The agent found no sufficient evidence of fraud, and therefore took **{action}**."
        )
    
    # Update the state with the generated explanation
    return {"explanation_report": explanation}


def _heuristic_fallback(event: str) -> dict:
    """Applies a simple, robust heuristic if the LLM fails."""
    ev = (event or "").lower()
    # Your original heuristic logic
    heur = "suspicious" in ev or ("director" in ev and "22" in ev)
    
    logger.warning(f"Using Heuristic Fallback for event: '{event[:20]}...'.")
    return {
        "suspicious_signup": heur, 
        "normal_signup": not heur,
        "confidence": 0.5,
        "reason": "LLM failed, used heuristic fallback."
    }

def _classify_with_openai(event: str) -> dict:
    """
    Classifies an event using the OpenAI API with structured output.
    Raises LLMClassificationError on failure or ambiguous classification.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMClassificationError("OPENAI_API_KEY missing.")
    
    try:
        user_prompt_content = PROMPT_CLASSIFY_USER.format(event=event)
        
        # We pass the system and user content as messages to the LLM
        result: Signals = structured.invoke([
            ("system", PROMPT_CLASSIFY_SYSTEM),
            ("user", user_prompt_content)
        ])
        signals = result.model_dump()

        # Sanity check: enforce exactly one True
        if signals.get("suspicious_signup") == signals.get("normal_signup"):
            raise LLMClassificationError("Ambiguous classification from LLM.")
            
        return signals

    except APIError as e:
        raise LLMClassificationError(f"OpenAI API error: {e}") from e
    
    except json.JSONDecodeError as e:
        raise LLMClassificationError(f"Invalid JSON received from LLM: {e}") from e
    
    except Exception as e:
        raise LLMClassificationError(f"Unexpected classification error: {type(e).__name__}: {e}") from e

# --- GRAPH NODES ---

@traceable(project_name=PROJECT, tags=["node"])
def user_event_node(state: GraphState) -> GraphState:
    print("--- 1. User Event Node ---")
    return {"event_data": state.get("event_data", ""), "signals": {}, "action": "", "log_entry": "", "classification": "error"}

@traceable(project_name=PROJECT, tags=["node"])
def signal_evaluation_node(state: GraphState) -> GraphState:
    print("--- 2. Signal Evaluation Node (LLM/Heuristic) ---")
    event = state.get("event_data", "")
    
    try:
        signals = _classify_with_openai(event)
    except LLMClassificationError as e:
        logger.error(f"LLM Classification failed: {e}")
        signals = _heuristic_fallback(event)
    
    # Determine classification string for routing
    classification = "suspicious" if signals.get("suspicious_signup") else "normal"
        
    return {"signals": signals, "classification": classification}

@traceable(project_name=PROJECT, tags=["node"])
def action_decision_node(state: GraphState) -> GraphState:
    print("--- 3. Action Decision Node ---")
    
    if state.get("classification") == "suspicious": 
        # Simulation: randomly choose a high-impact action
        action = "remove_account" if random.random() > 0.5 else "hold_account"
    else:
        action = "no_action"
        
    return {"action": action}

@traceable(project_name=PROJECT, tags=["node"])
def logging_node(state: GraphState) -> GraphState:
    print("--- 4. Logging Node ---")
    log_summary = {
        "event_summary": state.get("event_data", "N/A")[:50] + "...",
        "classification": state.get("classification"),
        "signals_detected": state.get("signals", {}),
        "final_action": state.get("action", "N/A"),
        "timestamp": datetime.now().isoformat(),
    }
    log_entry = json.dumps(log_summary, indent=2)
    print("\n--- Final Log Entry ---\n" + log_entry)
    return {"log_entry": log_entry}

# --- ROUTER FUNCTION ---

def route_to_action(state: GraphState) -> str:
    """Decides the next step based on the classification."""
    classification = state.get("classification")
    print(f"--- Router: Classification is '{classification}' ---")
    
    if classification == "suspicious":
        return "decide_action"
    
    # Skip decision and move straight to logging for 'normal' cases
    return "skip_action"

# --- GRAPH ASSEMBLY ---

def graph():
    wf = StateGraph(GraphState)
    
    # Define Nodes
    wf.add_node("event_input", user_event_node) 
    wf.add_node("classify_signals", signal_evaluation_node)
    wf.add_node("decide_action", action_decision_node)
    wf.add_node("log_result", logging_node)
    
    # Define Edges
    wf.add_edge(START, "event_input")
    wf.add_edge("event_input", "classify_signals")
    
    # Conditional Edge: The Performance Optimization
    wf.add_conditional_edges(
        "classify_signals",
        route_to_action, # The router function
        {
            "decide_action": "decide_action", # If suspicious, go to action node
            "skip_action": "log_result",       # If normal, skip action and go to log
        },
    )
    
    # Sequential Edges after the fork
    wf.add_edge("decide_action", "log_result") 
    wf.add_edge("log_result", END)             
    
    return wf.compile()
# --- EXAMPLE EXECUTION BLOCK (Updated to prompt for input) ---
if __name__ == "__main__":
    
    app = graph()

    # 1. Ensure project exists and send a ping for LangSmith tracing (optional but good practice)
    # The client creation is kept here to initialize tracing settings
    try:
        from langsmith import Client
        c = Client()
        c.create_project(PROJECT, upsert=True)
        print(f"✅ Agent ready. LangSmith Project: {PROJECT}")
    except Exception:
        # Fails silently if LANGSMITH_API_KEY is missing, but agent still runs
        print("✅ Agent ready. (LangSmith tracing disabled)")
        
    print("\n" + "="*50)
    print("PANEL MONITORING AGENT READY")
    print("Type a signup event description and press Enter (e.g., 'New user is a 22 year old director').")
    print("="*50)

    # Loop to continuously prompt for input
    while True:
        try:
            # Prompt the user for event data
            event = input("\nEvent Data > ")
            
            if not event.strip():
                print("Exiting agent.")
                break

            # Execute the LangGraph workflow
            # The run is wrapped in 'trace' for LangSmith tracking
            with trace(name=f"Manual Run - {datetime.now().strftime('%H:%M:%S')}", project_name=PROJECT):
                final_state = app.invoke({"event_data": event})

            print("\n--- AGENT RESULTS ---")
            
            # Print the new explanation first
            print("--- AGENT EXPLANATION ---")
            print(final_state.get('explanation_report', 'Explanation generation failed.'))
            
            print("\n--- FULL LOG DATA ---")
            print(f"Classification: {final_state.get('classification')}")
            print(f"Final Action : {final_state.get('action')}")
            print("--- Full Log ---")
            print(final_state.get('log_entry'))
            
        except KeyboardInterrupt:
            print("\nExiting agent.")
            break
        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred: {e}")
            break