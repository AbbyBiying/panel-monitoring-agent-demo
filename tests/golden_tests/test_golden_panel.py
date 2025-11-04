import json
import os
import pathlib
import pytest

from langgraph.types import Command

from panel_monitoring.app.graph import build_graph
from panel_monitoring.app.nodes import (
    CONFIDENCE_REVIEW_THRESHOLD,
    CONFIDENCE_UNCERTAIN_THRESHOLD,
)

# Stable deterministic test env
os.environ.setdefault("OPENAI_MODEL", "gpt-4o-mini")
os.environ.setdefault("LLM_TEMPERATURE", "0")
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("PANEL_DEFAULT_PROVIDER", "vertexai")
os.environ.setdefault("VERTEX_MODEL", "gemini-2.5-pro")

HERE = pathlib.Path(__file__).parent
TEST_DATA = HERE / "test-data.json"


def load_test_data():
    if not TEST_DATA.exists():
        raise FileNotFoundError(f"test-data.json not found at: {TEST_DATA.resolve()}")

    with open(TEST_DATA, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("test-data.json must be a JSON list.")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} in test-data.json is not an object.")
        for key in ("pid", "removed", "story"):
            if key not in item:
                raise KeyError(f"Item {i} missing required key: {key}")

    return data


@pytest.mark.parametrize("item", load_test_data()[:1])
def test_golden_panel(item):
    """
    Golden test that validates:
      - model classification aligns with expected removed flag
      - HITL auto-resolution path works for tests
      - explanation + logging always populated for confident decisions
    """

    thread_id = f"golden:{item['pid']}"
    cfg = {"configurable": {"thread_id": thread_id}}
    app = build_graph()

    state = app.invoke({"event_data": item}, config=cfg)

    # --- Auto-resolve HITL if triggered ---
    if state.get("__interrupt__"):
        expected_removed = bool(item["removed"])
        decision = "approve" if expected_removed else "reject"

        state = app.invoke(Command(resume=decision), config=cfg)

    # --- Extract results ---
    classification = (state.get("classification") or "").lower()
    confidence = float(state.get("confidence") or 0.0)
    action = state.get("action") or ""
    explanation = state.get("explanation_report") or ""

    expected_removed = bool(item["removed"])
    actual_removed = None
    # --- Golden label logic ---
    if classification in ("suspicious", "error") and confidence > 0.30:
        actual_removed = True
    elif classification == "normal" and confidence > 0.40:
        actual_removed = False

    if actual_removed is not None:
        assert actual_removed == expected_removed, (
            f"pid={item['pid']} expected_removed={expected_removed} "
            f"got classification={classification} conf={confidence:.2f}"
        )

    # If HITL should have happened, ensure we ended with a final action
    if confidence >= CONFIDENCE_REVIEW_THRESHOLD:
        assert action in {"delete_account", "hold_account", "no_action"}

    # Explanation must not be empty for confident decisions or final actions
    if confidence >= CONFIDENCE_UNCERTAIN_THRESHOLD or action in {
        "delete_account",
        "hold_account",
    }:
        assert explanation.strip(), "explanation_report should not be empty"

    # --- Validate logging ---
    log_entry = state.get("log_entry")
    assert isinstance(log_entry, str) and log_entry.strip(), (
        "log_entry missing or empty"
    )

    # Light structural validation of log fields (avoid full parse)
    for k in ("event_id", "classification", "confidence", "provider", "model"):
        assert k in log_entry, f"log_entry missing '{k}'"