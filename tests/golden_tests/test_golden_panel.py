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
TEST_DATA = HERE / "cleaned_test.json"


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
PARAM_ITEMS = load_test_data()[0:2]


@pytest.mark.parametrize("item", PARAM_ITEMS)
def test_golden_panel(item):
    """
    Golden test:
      - ASSERT final classification vs expected `ground_truth.removed`.
      - Do not let HITL hide misclassifications; we assert on the final classification.
      - Keep minimal structural assertions (confidence/explanation/logging).
    """
    pid = item["id"]
    gt = item["ground_truth"]

    thread_id = f"golden:{pid}"
    cfg = {"configurable": {"thread_id": thread_id}}
    app = build_graph()

    event_payload = _build_event_payload(item)
    state = app.invoke({"event_data": event_payload}, config=cfg)

    # If HITL (interrupt) happens, resume using expected truth,
    # but we still assert on FINAL classification.
    if state.get("__interrupt__"):
        # If we *expect* the account to be removed, human should "approve" deletion.
        # Otherwise, they should "reject" it.
        decision = "approve" if bool(gt["removed"]) else "reject"
        state = app.invoke(Command(resume=decision), config=cfg)

    # ---- Extract final fields ----
    classification = (state.get("classification") or "").strip().lower()
    confidence = float(state.get("confidence") or 0.0)
    action = (state.get("action") or "").strip()
    explanation = (state.get("explanation_report") or "").strip()
    signals = state.get("signals") or {}
    reason = signals.get("reason") if isinstance(signals, dict) else getattr(
        signals, "reason", ""
    )

    print(f"\n[RESULT] pid={pid} class={classification} conf={confidence:.2f} action={action}")
    if reason:
        print("[REASON]", reason)
    if explanation:
        print("[EXPLANATION]", explanation[:400], "..." if len(explanation) > 400 else "")


    class_to_removed = {"suspicious": True, "normal": False}
    assert classification in class_to_removed, f"Unknown classification: {classification!r}"

    expected_removed = bool(gt["removed"])
    actual_removed = class_to_removed[classification]
    assert actual_removed == expected_removed, (
        f"pid={pid} expected_removed={expected_removed} "
        f"but classification={classification} (conf={confidence:.2f}, action={action})"
    )

    # ---- Sanity: explanation/logging must exist for review+ confidence or final actions ----
    if confidence >= CONFIDENCE_UNCERTAIN_THRESHOLD or action in {
        "delete_account",
        "hold_account",
    }:
        assert explanation, "explanation_report should not be empty"

    log_entry = state.get("log_entry")
    assert isinstance(log_entry, str) and log_entry.strip(), "log_entry missing or empty"
    for k in ("event_id", "classification", "confidence", "provider", "model"):
        assert k in log_entry, f"log_entry missing '{k}'"
