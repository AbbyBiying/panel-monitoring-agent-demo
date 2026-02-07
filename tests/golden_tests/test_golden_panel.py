# tests/test_golden_panel.py
import json
import os
import pathlib
import pytest
from typing import Any, Dict

from langgraph.types import Command
from panel_monitoring.app.graph import build_graph

# Stable env for CI
os.environ.setdefault("OPENAI_MODEL", "gpt-4o-mini")
os.environ.setdefault("LLM_TEMPERATURE", "0")
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("PANEL_DEFAULT_PROVIDER", "vertexai")
os.environ.setdefault("VERTEX_MODEL", "gemini-2.5-flash")

HERE = pathlib.Path(__file__).parent
TEST_DATA = HERE / "formatted-test-data.json"
# TEST_DATA = HERE / "formatted-edge-test-data.json"


def _get_pid(item: Dict[str, Any]) -> str:
    """Extract panelist_id from the item's identity block."""
    return item.get("identity", {}).get("panelist_id", "unknown")


def _build_event_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the event_data payload from a golden test item.

    We pass everything except ground_truth as event_data. The graph will
    turn this into event_text internally via json.dumps(event_data).
    """
    return {k: v for k, v in item.items() if k != "ground_truth"}


def load_test_data() -> list[Dict[str, Any]]:
    """
    Expect test data JSON in this format (no 'id' or 'input' wrapper):

    [
      {
        "identity": { "panelist_id": "iP7bb1447733", ... },
        "network_device": { ... },
        ...
        "ground_truth": {
          "removed": false,
          "label": "normal",
          "action": "no_action"
        }
      },
      ...
    ]
    """
    if not TEST_DATA.exists():
        raise FileNotFoundError(
            f"Test data not found at: {TEST_DATA.resolve()}"
        )

    with open(TEST_DATA, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Test data must be a JSON list.")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} in test data is not an object.")

        if "ground_truth" not in item:
            raise KeyError(f"Item {i} missing required key: 'ground_truth'")

        if not isinstance(item["ground_truth"], dict):
            raise TypeError(
                f"Item {i} 'ground_truth' must be a JSON object (dict), "
                f"got {type(item['ground_truth'])}."
            )

        # Minimal ground truth contract
        gt = item["ground_truth"]
        if "removed" not in gt:
            raise KeyError(f"Item {i} ground_truth missing 'removed' key.")
        if not isinstance(gt["removed"], bool):
            raise TypeError(
                f"Item {i} ground_truth['removed'] must be bool, "
                f"got {type(gt['removed'])}."
            )

    return data


def calculate_human_signal_score(item: Dict[str, Any]) -> int:
    """Heuristic to track if the data looks human before the Agent sees it."""
    score = 0
    signals = item.get("third_party_signals", {})

    # 1. Email Age
    email_date = signals.get("email_first_seen_online", "2026-01-01")
    try:
        year = int(email_date.split("-")[0])
        if year <= 2020:
            score += 40
        elif year <= 2023:
            score += 20
    except Exception:
        pass

    # 2. MinFraud (Low risk is high human signal)
    try:
        risk = float(signals.get("minfraud_risk_score", 100))
        if risk < 1.0:
            score += 30
        elif risk < 10.0:
            score += 15
    except Exception:
        pass

    # 3. reCAPTCHA
    try:
        captcha = float(signals.get("recaptcha_score", 0))
        if captcha >= 0.7:
            score += 30
    except Exception:
        pass

    return min(score, 100)


# Change slice if you want to run a subset: e.g., [:1], [1:2], etc.
# PARAM_ITEMS = load_test_data()[0:2]
PARAM_ITEMS = load_test_data()
#i only want to test iP1c42f07cff, iP2d8325efb9, iP5a77f99280, iPbd9fc018e8
# PARAM_ITEMS = [item for item in PARAM_ITEMS if item["identity"]["panelist_id"] in ["iP1c42f07cff", "iP2d8325efb9", "iP5a77f99280", "iPbd9fc018e8"]]


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("item", PARAM_ITEMS, ids=[_get_pid(it) for it in PARAM_ITEMS])
async def test_golden_panel(item):
    pid = _get_pid(item)
    gt = item["ground_truth"]
    thread_id = f"golden:{pid}"
    cfg = {"configurable": {"thread_id": thread_id}}
    app = build_graph()

    signal_strength = calculate_human_signal_score(item)
    state = await app.ainvoke({"event_data": _build_event_payload(item)}, config=cfg)

    # Handle Human-in-the-loop if triggered
    if state.get("__interrupt__"):
        decision = "approve" if not bool(gt["removed"]) else "reject"
        state = await app.ainvoke(Command(resume=decision), config=cfg)

    # Extract Agent Results
    classification = (state.get("classification") or "").upper()
    confidence = float(state.get("confidence") or 0.0)
    signals_obj = state.get("signals")

    # ---- CLEAN CONSOLE OUTPUT ----
    print(f"\nID: {pid} | VERDICT: {classification} ({confidence * 100:.0f}%)")
    print(f"  [SIGNAL STRENGTH: {signal_strength}/100 | GT_REMOVED: {gt['removed']}]")

    print("-" * 40)

    # ---- ERROR ANALYSIS & ASSERTIONS ----
    class_to_removed = {"SUSPICIOUS": True, "NORMAL": False}
    actual_removed = class_to_removed.get(classification, False)

    # Check for Mismatch
    is_mismatch = actual_removed != bool(gt["removed"])

    # --- AGE BIAS DETECTION ---
    # Extract age from registration_profile
    raw_age = item.get("registration_profile", {}).get("age", 0)
    try:
        user_age = int(raw_age)
    except (ValueError, TypeError):
        user_age = 0
    is_senior = user_age >= 55

    if is_mismatch:
        # If it's a False Positive on a Senior
        if is_senior and actual_removed is True and not bool(gt["removed"]):
            print(
                f"      BIAS ALERT: Potential Age-Bias detected for ID {pid} (Age: {user_age})."
            )
            print("      The Agent flagged this senior despite 'Normal' Ground Truth.")
            print(f"      Signal Strength: {signal_strength}/100")

        reason = (
            signals_obj.get("reason") if isinstance(signals_obj, dict)
            else getattr(signals_obj, "reason", "N/A")
        )
        msg = (
            f"\n   Mismatch on {pid}:"
            f"\n   User Persona: {'Senior' if is_senior else 'General'} (Age: {user_age})"
            f"\n   Agent Verdict: {classification}"
            f"\n   Ground Truth: {'REJECT' if gt['removed'] else 'APPROVE'}"
            f"\n   Human Signal: {signal_strength}/100"
            f"\n   Reason: {reason}"
        )
        assert not is_mismatch, msg
    pid = _get_pid(item)
    gt = item["ground_truth"]
    thread_id = f"golden:{pid}"
    cfg = {"configurable": {"thread_id": thread_id}}
    app = build_graph()

    # Calculate our "Ground Truth" signal strength before running the agent
    signal_strength = calculate_human_signal_score(item)

    state = await app.ainvoke({"event_data": _build_event_payload(item)}, config=cfg)

    if state.get("__interrupt__"):
        decision = "approve" if bool(gt["removed"]) else "reject"
        state = await app.ainvoke(Command(resume=decision), config=cfg)

    # ---- Minimal Extraction ----
    classification = (state.get("classification") or "").upper()
    confidence = float(state.get("confidence") or 0.0)
    signals = state.get("signals")

    # ---- DEBUG OUTPUT ----
    print(f"\nID: {pid} | VERDICT: {classification} ({confidence * 100:.0f}%)")
    print(f"  [SIGNAL STRENGTH: {signal_strength}/100 | GT_REMOVED: {gt['removed']}]")

    reason = (
        signals.get("reason") if isinstance(signals, dict)
        else getattr(signals, "reason", "N/A")
    )
    print(f"  Reason: {reason}")

    print("-" * 40)

    # ---- Assertions (Silent unless they fail) ----
    class_to_removed = {"SUSPICIOUS": True, "NORMAL": False}
    actual_removed = class_to_removed.get(classification, False)
    msg = (
        f"\nFAILED: {pid}"
        f"\nAgent said: {classification}"
        f"\nGround Truth Removed: {gt['removed']}"
        f"\nHuman Signal Strength was {signal_strength}."
        f"\nReason: {reason if reason else 'N/A'}"
    )
    assert actual_removed == bool(gt["removed"]), f"Mismatch on {pid}"
    if actual_removed != bool(gt["removed"]):
        if signal_strength > 60 and actual_removed:
            print(
                f"  !!! ALERT: Agent is being too paranoid. Signal Strength is high ({signal_strength})."
            )

        assert actual_removed == bool(gt["removed"]), msg
