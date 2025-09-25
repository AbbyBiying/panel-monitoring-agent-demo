# panel_agent.py
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import os, json, random

# ---------- State ----------
class GraphState(TypedDict):
    event_data: str
    signals: dict
    action: str
    log_entry: str

# ---------- OpenAI helper ----------
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _classify_with_openai(event: str) -> dict:
    """
    Return {"suspicious_signup": bool, "normal_signup": bool}
    Falls back to heuristic if OpenAI isn't available or errors.
    """
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        client = OpenAI(api_key=api_key)

        prompt = (
            "You are a JSON-only classifier for signup events.\n"
            "Return exactly this JSON schema:\n"
            '{ "suspicious_signup": <bool>, "normal_signup": <bool> }\n'
            "Exactly one must be true.\n\n"
            f"Event: {event}"
        )

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        suspicious = bool(data.get("suspicious_signup", False))
        normal = bool(data.get("normal_signup", False))

        # sanity: enforce exactly one True
        if suspicious == normal:
            raise ValueError("Ambiguous JSON, using heuristic")
        return {"suspicious_signup": suspicious, "normal_signup": normal}

    except Exception as e:
        # Heuristic fallback if model/env not available or JSON invalid
        ev = (event or "").lower()
        heur = "suspicious" in ev or ("director" in ev and "22" in ev)
        return {"suspicious_signup": heur, "normal_signup": not heur}

# ---------- Nodes ----------
def user_event_node(state: GraphState) -> GraphState:
    print("--- 1. User Event Node ---")
    return {"event_data": state.get("event_data", ""), "signals": {}, "action": "", "log_entry": ""}

def signal_evaluation_node(state: GraphState) -> GraphState:
    print("--- 2. Signal Evaluation Node (OpenAI) ---")
    event = state.get("event_data", "")
    signals = _classify_with_openai(event)
    return {"signals": signals}

def action_decision_node(state: GraphState) -> GraphState:
    print("--- 3. Action Decision Node ---")
    sigs = state.get("signals", {})
    if sigs.get("suspicious_signup"):
        action = "remove_account" if random.random() > 0.5 else "hold_account"
    else:
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
