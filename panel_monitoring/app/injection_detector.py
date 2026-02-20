# panel_monitoring/app/injection_detector.py
"""
DeBERTa-based prompt injection detector.

Uses protectai/deberta-v3-base-prompt-injection-v2 to classify text as
INJECTION or SAFE. The model is loaded lazily on first use and cached
as a singleton.

This serves as the second detection layer after the lightweight regex
scanner in utils.detect_prompt_injection().
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass

from panel_monitoring.app.retry import llm_retry

logger = logging.getLogger(__name__)

_MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"
_pipeline = None
_load_lock = threading.Lock()


@dataclass(frozen=True)
class ModelScanResult:
    """Result from the DeBERTa injection model."""
    label: str          # "INJECTION" or "SAFE"
    score: float        # confidence score [0, 1]
    detected: bool      # True if label == "INJECTION"
    source: str = ""    # "event" or "retrieved_doc_N"


def _get_pipeline():
    """Lazy-load the HuggingFace pipeline (thread-safe singleton)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    with _load_lock:
        if _pipeline is not None:
            return _pipeline

        from transformers import pipeline  # noqa: E402

        logger.info("Loading prompt injection model: %s", _MODEL_NAME)
        _pipeline = pipeline(
            "text-classification",
            model=_MODEL_NAME,
            truncation=True,
            max_length=512,
        )
        logger.info("Prompt injection model loaded successfully.")
        return _pipeline


@llm_retry
def _classify_sync(text: str) -> dict:
    """Run the model synchronously. Returns {"label": ..., "score": ...}."""
    pipe = _get_pipeline()
    result = pipe(text)
    # pipeline returns [{"label": "INJECTION", "score": 0.99}]
    if not result:
        raise RuntimeError("Injection detection pipeline returned empty result")
    return result[0]


async def detect_injection_ml(
    text: str,
    *,
    source: str = "event",
    threshold: float = 0.5,
) -> ModelScanResult:
    """
    Async-safe DeBERTa prompt injection detection.

    Offloads the blocking model inference to a thread pool
    (same pattern as the LLM clients).

    Args:
        text: The untrusted text to scan.
        source: Label for audit trail ("event", "retrieved_doc_1", etc.).
        threshold: Score above which INJECTION label triggers detection.

    Returns:
        ModelScanResult with label, score, and detected flag.
    """
    if not text or not text.strip():
        return ModelScanResult(
            label="SAFE", score=0.0, detected=False, source=source,
        )

    raw = await asyncio.to_thread(_classify_sync, text)
    label = raw.get("label", "SAFE")
    score = float(raw.get("score", 0.0))
    detected = label == "INJECTION" and score >= threshold

    if detected:
        logger.warning(
            "ML injection detected | source=%s | label=%s | score=%.4f | text_preview=%.200s",
            source, label, score, text[:200],
        )

    return ModelScanResult(
        label=label,
        score=score,
        detected=detected,
        source=source,
    )
