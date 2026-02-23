# services/deberta-api/main.py
"""FastAPI inference service for DeBERTa v3 prompt injection classifier.

Run locally:
  just serve-injection-api

Or from repo root:
  uv run uvicorn main:app --reload --app-dir services/deberta-api --port 8080
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager

# Ensure repo root is on path so panel_monitoring is importable
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from fastapi import FastAPI
from pydantic import BaseModel, Field

from panel_monitoring.app.injection_detector import detect_injection_ml

_MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the model at startup so the first request isn't slow."""
    await detect_injection_ml("warmup", source="startup")
    yield


app = FastAPI(
    title="DeBERTa Injection Classifier",
    description="Prompt injection detection inference service. Wraps protectai/deberta-v3-base-prompt-injection-v2.",
    lifespan=lifespan,
)


class ClassifyRequest(BaseModel):
    text: str
    source: str = Field(default="event")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    detected: bool
    source: str
    model: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model=_MODEL_NAME)


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    t0 = time.perf_counter()
    result = await detect_injection_ml(
        req.text,
        source=req.source,
        threshold=req.threshold,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return ClassifyResponse(
        label=result.label,
        confidence=result.score,
        detected=result.detected,
        source=result.source,
        model=_MODEL_NAME,
        latency_ms=latency_ms,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
