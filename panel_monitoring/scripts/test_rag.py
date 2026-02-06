# panel_monitoring/scripts/test_rag.py
"""
Quick smoke test for the RAG pipeline.
Tests retrieval_node independently, then runs a full graph invocation.

Usage:
  uv run python -m panel_monitoring panel_monitoring/scripts/test_rag.py
"""

from __future__ import annotations

import asyncio
import json
import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Sample suspicious event for testing
SAMPLE_EVENT = {
    "identity": {
        "panelist_id": "TEST-RAG-001",
        "primary_email_domain": "tempmail.com",
        "email_first_seen_online": "2025-01-15",
        "year_of_birth": 1990,
        "age": 35,
        "occupation": "student",
    },
    "signup": {
        "source": "organic",
        "ip_country": "US",
        "ip_state": "CA",
        "claimed_state": "NY",
        "user_agent": "Mozilla/5.0",
    },
    "flags": {
        "minfraud_risk_score": 25.0,
        "same_ip_as_existing_user": True,
        "recaptcha_score": 0.3,
        "vpngate_user": False,
    },
}


async def test_retrieval_only():
    """Test just the embedding + vector search step."""
    from panel_monitoring.data.firestore_client import embed_text, get_similar_patterns

    event_text = json.dumps(SAMPLE_EVENT, ensure_ascii=False)
    logger.info("=== Testing Retrieval Only ===")
    logger.info("Embedding event text (%d chars)...", len(event_text))

    query_vector = await embed_text(event_text)
    logger.info("Embedding dim: %d", len(query_vector))

    docs = await get_similar_patterns(query_vector, limit=5)
    logger.info("Retrieved %d documents:", len(docs))
    for i, doc in enumerate(docs, 1):
        text_preview = doc.get("text", "")[:120]
        logger.info("  [%d] id=%s | %s...", i, doc.get("id"), text_preview)

    return docs


async def test_full_graph():
    """Test the full graph with RAG retrieval wired in."""
    from uuid import uuid4
    from panel_monitoring.app.graph import build_graph

    logger.info("\n=== Testing Full Graph with RAG ===")
    app = build_graph()

    event_id = uuid4().hex
    payload = {
        "event_id": event_id,
        "event_data": SAMPLE_EVENT,
    }
    config = {"configurable": {"thread_id": event_id}}

    result = await app.ainvoke(payload, config=config)

    logger.info("\n--- Result ---")
    logger.info("classification: %s", result.get("classification"))
    logger.info("confidence: %s", result.get("confidence"))
    logger.info("retrieved_docs count: %d", len(result.get("retrieved_docs", [])))

    for i, doc in enumerate(result.get("retrieved_docs", []), 1):
        logger.info("  retrieved[%d]: %s...", i, doc.get("text", "")[:100])

    signals = result.get("signals")
    if signals:
        sig = signals if isinstance(signals, dict) else signals.model_dump()
        logger.info("signals.reason: %s", sig.get("reason", ""))

    logger.info("\nexplanation_report: %s", result.get("explanation_report"))
    logger.info("\nlog_entry preview: %s", (result.get("log_entry") or "")[:300])


async def main():
    docs = await test_retrieval_only()
    if not docs:
        logger.error("No documents retrieved! Check that ingestion ran successfully.")
        return

    await test_full_graph()
    logger.info("\n=== RAG Test Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
