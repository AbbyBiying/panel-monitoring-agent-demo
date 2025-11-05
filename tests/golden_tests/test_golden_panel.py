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
    if not TEST_DATA.exists():
        raise FileNotFoundError(f"cleaned_test.json not found at: {TEST_DATA.resolve()}")

    with open(TEST_DATA, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("cleaned_test.json must be a JSON list.")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} in cleaned_test.json is not an object.")
        for key in ("pid", "removed", "story"):
            if key not in item:
                raise KeyError(f"Item {i} missing required key: {key}")
        if not isinstance(item["story"], (str, dict)):
            raise TypeError(f"Item {i} 'story' must be str or dict, got {type(item['story'])}.")

    return data


def _normalize_event(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize story -> story_text for consistent LLM input."""
    story = item.get("story")
    event = dict(item)  # shallow copy
    if isinstance(story, dict):
        event["story_text"] = json.dumps(story, ensure_ascii=False, separators=(", ", ": "), indent=2)
    else:
        event["story_text"] = "\n".join(line.strip() for line in str(story).splitlines()).strip()
    return event


# Change slice if you want to run a subset: e.g., [:1], [1:2], etc.
PARAM_ITEMS = load_test_data()[1:2]


@pytest.mark.parametrize("item", PARAM_ITEMS)
def test_golden_panel(item):
    """
    Golden test (Option B):
      - ASSERT classification-only vs expected `removed` label.
      - Do not let HITL hide misclassifications; we assert on the final classification.
      - Keep minimal structural assertions (confidence/explanation/logging).
    """
    thread_id = f"golden:{item['pid']}"
    cfg = {"configurable": {"thread_id": thread_id}}
    app = build_graph()

    event_payload = _normalize_event(item)
    state = app.invoke({"event_data": event_payload}, config=cfg)

    # If HITL (interrupt) happens, resume using expected truth, but we still assert on FINAL classification.
    if state.get("__interrupt__"):
        decision = "approve" if bool(item["removed"]) else "reject"
        state = app.invoke(Command(resume=decision), config=cfg)

    # ---- Extract final fields ----
    classification = (state.get("classification") or "").strip().lower()
    confidence = float(state.get("confidence") or 0.0)
    action = (state.get("action") or "").strip()
    explanation = (state.get("explanation_report") or "").strip()
    signals = state.get("signals") or {}
    reason = signals.get("reason") if isinstance(signals, dict) else getattr(signals, "reason", "")

    print(f"\n[RESULT] pid={item['pid']} class={classification} conf={confidence:.2f} action={action}")
    if reason:
        print("[REASON]", reason)
    if explanation:
        print("[EXPLANATION]", explanation[:400], "..." if len(explanation) > 400 else "")

    # ---- STRICT assertion: classification -> removed mapping ----
    # Map final model classification to a boolean "removed" truth
    class_to_removed = {"suspicious": True, "normal": False}
    assert classification in class_to_removed, f"Unknown classification: {classification!r}"

    expected_removed = bool(item["removed"])
    actual_removed = class_to_removed[classification]
    assert actual_removed == expected_removed, (
        f"pid={item['pid']} expected_removed={expected_removed} "
        f"but classification={classification} (conf={confidence:.2f}, action={action})"
    )

    # ---- Sanity: explanation/logging must exist for review+ confidence or final actions ----
    if confidence >= CONFIDENCE_UNCERTAIN_THRESHOLD or action in {"delete_account", "hold_account"}:
        assert explanation, "explanation_report should not be empty"

    # ---- Light logging validation (string presence + key hints) ----
    log_entry = state.get("log_entry")
    assert isinstance(log_entry, str) and log_entry.strip(), "log_entry missing or empty"
    for k in ("event_id", "classification", "confidence", "provider", "model"):
        assert k in log_entry, f"log_entry missing '{k}'"
