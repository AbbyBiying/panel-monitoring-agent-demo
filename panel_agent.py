# panel_agent.py
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import os, json, random
from typing import Literal, TypedDict
from openai import OpenAI, APIError # Import necessary classes outside the function
import os, json, random


PROMPT_CLASSIFY_SYSTEM = (
    "You are a dedicated fraud and abuse detection expert for a consumer survey panel. "
    "Your only function is to classify user signup events. ALWAYS return only the JSON schema provided."
)

PROMPT_CLASSIFY_USER = """
Analyze the following panelist event data for signals of fraud or abuse.

1.  Return the classification in the provided JSON schema.
2.  "suspicious_signup" must be TRUE if the event shows characteristics of bots, identity theft, or policy abuse (e.g., using a disposable email, rapid-fire activity, or using known fraudulent keywords).
3.  "normal_signup" must be TRUE ONLY if the event appears to be from a legitimate, unique user.
4.  Exactly one of the two must be TRUE.

Event Data: {event}
"""

# ---------- State ----------
class GraphState(TypedDict):
    event_data: str
    signals: dict 
    classification: Literal["suspicious", "normal", "error"] # Added for clean routing
    action: str
    log_entry: str

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

class LLMClassificationError(Exception):
    """Raised when the LLM response is invalid, ambiguous, or fails."""
    pass


def _classify_with_openai(event: str) -> dict:
    """
    Classifies an event using the OpenAI API.
    
    Raises:
        LLMClassificationError: If API communication or response validation fails.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMClassificationError("OPENAI_API_KEY missing from environment variables.")
    
    try:
        client = OpenAI(api_key=api_key)
        
        # 1. Construct the final prompt using the templates
        user_prompt_content = PROMPT_CLASSIFY_USER.format(event=event)

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                # System prompt sets the persona and primary rule
                {"role": "system", "content": PROMPT_CLASSIFY_SYSTEM}, 
                # User prompt provides the data and detailed instructions
                {"role": "user", "content": user_prompt_content},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            timeout=15 # Added timeout for robustness
        )
        
        content = resp.choices[0].message.content
        data = json.loads(content)
        suspicious = bool(data.get("suspicious_signup", False))
        normal = bool(data.get("normal_signup", False))

        # 2. Sanity check: enforce exactly one True
        if suspicious == normal:
            raise LLMClassificationError(
                f"Ambiguous JSON from LLM: 'suspicious' and 'normal' are both {suspicious}. Raw content: {content}"
            )
            
        return {"suspicious_signup": suspicious, "normal_signup": normal}

    except APIError as e:
        raise LLMClassificationError(f"OpenAI API error: {e}") from e
    
    except json.JSONDecodeError as e:
        raise LLMClassificationError(f"Invalid JSON received from LLM: {e}") from e
    
    except Exception as e:
        raise LLMClassificationError(f"Unexpected classification error: {type(e).__name__}: {e}") from e

# ---------- Nodes ----------
def user_event_node(state: GraphState) -> GraphState:
    print("--- 1. User Event Node ---")
    return {"event_data": state.get("event_data", ""), "signals": {}, "action": "", "log_entry": ""}

def signal_evaluation_node(state: GraphState) -> GraphState:
    print("--- 2. Signal Evaluation Node (OpenAI) ---")
    event = state.get("event_data", "")
    signals = _classify_with_openai(event)
      # Determine classification string for routing
    if signals.get("suspicious_signup"):
        classification = "suspicious"
    elif signals.get("normal_signup"):
        classification = "normal"
    else:
        # If the heuristic failed or returned an ambiguous result
        classification = "error" 
        
    return {"signals": signals, "classification": classification}


# --- 3. Action Decision Node (No change in internal logic) ---
def action_decision_node(state: GraphState) -> GraphState:
    print("--- 3. Action Decision Node ---")
    sigs = state.get("signals", {})
    
    # Note: Use a single source of truth for the action logic
    if state.get("classification") == "suspicious": 
        # Keep your random action for simulation
        action = "remove_account" if random.random() > 0.5 else "hold_account"
    else:
        # Explicitly set to 'no_action' for normal/error cases
        action = "no_action"
        
    return {"action": action}


def logging_node(state: GraphState) -> GraphState:
    print("--- 4. Logging Node ---")
    log = {
        "event": state.get("event_data", "N/A"),
        "signals_detected": state.get("signals", {}),
        "final_action": state.get("action", "N/A"),
    }
    entry = json.dumps(log, indent=2)
    print("\n--- Final Log Entry ---\n" + entry)
    return {"log_entry": entry}

# ---------- Build graph ----------
def graph():
    wf = StateGraph(GraphState)
    wf.add_node("user_event_node", user_event_node)
    wf.add_node("signal_evaluation_node", signal_evaluation_node)
    wf.add_node("action_decision_node", action_decision_node)
    wf.add_node("logging_node", logging_node)
    wf.add_edge(START, "user_event_node")
    wf.add_edge("user_event_node", "signal_evaluation_node")
    wf.add_edge("signal_evaluation_node", "action_decision_node")
    wf.add_edge("action_decision_node", "logging_node")
    wf.add_edge("logging_node", END)
    return wf.compile()
