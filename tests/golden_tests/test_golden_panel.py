import csv
import os
import pathlib

import pytest


# Make tests stable
os.environ.setdefault("OPENAI_MODEL", "gpt-4o-mini")
os.environ.setdefault("LLM_TEMPERATURE", "0")
os.environ.setdefault("LANGSMITH_TRACING", "false")
from panel_monitoring.panel_agent_vertexai import graph

ROOT = pathlib.Path(__file__).parent
CSV = ROOT / "panel_agent_golden_tests_expanded.csv"


def load_golden_rows():
    if not CSV.exists():
        raise FileNotFoundError(f"Golden CSV not found at: {CSV.resolve()}")
    with open(CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("row", load_golden_rows())
def test_golden_panel(row):
    """Runs the full LangGraph pipeline so you also validate routing & logging."""
    g = graph()  # compiled graph from your code

    # Prepare event input (exactly what your CLI would receive)
    event_text = row["input"]

    # Run the agent
    final_state = g.invoke({"event_data": event_text})

    # Extract results
    classification = final_state.get("classification")
    signals = final_state.get("signals") or {}
    reason = signals.get("reason", "") or ""
    confidence = signals.get("confidence", 0.0) or 0.0

    # Assertions with tolerance
    assert classification == row["expected_classification"], (
        f"case={row['case_id']} expected={row['expected_classification']} got={classification}"
    )

    min_conf = float(row["min_confidence"] or 0.0)
    assert confidence >= min_conf, (
        f"case={row['case_id']} min_conf={min_conf} got={confidence}"
    )

    if row.get("reason_contains"):
        assert row["reason_contains"].lower() in reason.lower(), (
            f"case={row['case_id']} reason should contain '{row['reason_contains']}', got: {reason}"
        )
