# tests/test_golden_panel.py
import json
import os
import pathlib
import pytest
from typing import Any, Dict

from langgraph.types import Command
from panel_monitoring.app.graph import build_graph
from panel_monitoring.app.nodes import (
    CONFIDENCE_REVIEW_THRESHOLD,
    CONFIDENCE_UNCERTAIN_THRESHOLD,
)

# Stable env for CI
os.environ.setdefault("OPENAI_MODEL", "gpt-4o-mini")
os.environ.setdefault("LLM_TEMPERATURE", "0")
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("PANEL_DEFAULT_PROVIDER", "vertexai")
os.environ.setdefault("VERTEX_MODEL", "gemini-2.5-pro")

HERE = pathlib.Path(__file__).parent
TEST_DATA = HERE / "formatted-test-data.json"
# TEST_DATA = HERE / "formatted-edge-test-data.json"

def load_test_data() -> list[Dict[str, Any]]:
    """
    Expect cleaned_test.json in this format:

    [
      {
        "id": "iP7bb1447733",
        "input": { ... grouped identity/network/profile/demographics ... },
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
        raise FileNotFoundError(f"cleaned_test.json not found at: {TEST_DATA.resolve()}")

    with open(TEST_DATA, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("cleaned_test.json must be a JSON list.")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} in cleaned_test.json is not an object.")

        for key in ("id", "input", "ground_truth"):
            if key not in item:
                raise KeyError(f"Item {i} missing required key: {key!r}")

        if not isinstance(item["input"], dict):
            raise TypeError(
                f"Item {i} 'input' must be a JSON object (dict), "
                f"got {type(item['input'])}."
            )

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


def _build_event_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the event_data payload from a golden test item.

    We pass the grouped input (identity/network/profile/demographics)
    directly as event_data. The graph will turn this into event_text internally
    via _event_text_from_state -> json.dumps(event_data).
    """
    return item["input"]


# Change slice if you want to run a subset: e.g., [:1], [1:2], etc.
# PARAM_ITEMS = load_test_data()[0:2]
PARAM_ITEMS = load_test_data()


@pytest.mark.parametrize("item", PARAM_ITEMS,
    ids=[it["id"] for it in PARAM_ITEMS])
def test_golden_panel(item):
    pid = item["id"]
    gt = item["ground_truth"]
    thread_id = f"golden:{pid}"
    cfg = {"configurable": {"thread_id": thread_id}}
    app = build_graph()

    state = app.invoke({"event_data": _build_event_payload(item)}, config=cfg)

    if state.get("__interrupt__"):
        decision = "approve" if bool(gt["removed"]) else "reject"
        state = app.invoke(Command(resume=decision), config=cfg)

    # ---- Minimal Extraction ----
    classification = (state.get("classification") or "").upper()
    confidence = float(state.get("confidence") or 0.0)
    signals = state.get("signals")

    # ---- The "Clean" Output for the Meeting ----
    print(f"\nID: {pid} | VERDICT: {classification} ({confidence*100:.0f}%)")
    
    steps = []
    if isinstance(signals, dict):
        steps = signals.get("analysis_steps", [])
    else:
        steps = getattr(signals, "analysis_steps", [])

    if steps:
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
    else:
        reason = signals.get("reason") if isinstance(signals, dict) else getattr(signals, "reason", "N/A")
        print(f"  Note: {reason}")
    
    print("-" * 40)

    # ---- Assertions (Silent unless they fail) ----
    class_to_removed = {"SUSPICIOUS": True, "NORMAL": False}
    actual_removed = class_to_removed.get(classification, False)
    assert actual_removed == bool(gt["removed"]), f"Mismatch on {pid}"