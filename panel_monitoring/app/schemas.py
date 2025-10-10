# panel_monitoring/app/schemas.py

from typing import Literal, TypedDict, Dict, Any, Optional
from pydantic import BaseModel, Field


# ----------------------------
# Structured schema for LLM output
# ----------------------------

class Signals(BaseModel):
    suspicious_signup: bool = Field(..., description="True if the event is suspicious")
    normal_signup: bool = Field(..., description="True if the event is normal")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0,1]")
    reason: str = Field(..., description="Explanation or rationale for the classification")


# ----------------------------
# Metadata about model execution
# ----------------------------

class ModelMeta(TypedDict, total=False):
    provider: str
    model: str
    temperature: int
    max_output_tokens: int
    request_timeout: int
    max_retries: int
    usage: Dict[str, Any]
    latency_ms: int
    cost_usd: float
    error: str


# ----------------------------
# LangGraph state object
# ----------------------------

class GraphState(TypedDict, total=False):
    # Core identifiers
    project_id: str
    event_id: str

    # Input / raw content
    event_text: str                     # raw user/event input
    event_data: Dict[str, Any]          # structured input payload

    # Classification results
    signals: Dict[str, Any]             # normalized Signals
    classification: Literal["suspicious", "normal", "error"]
    confidence: float
    model_meta: ModelMeta
    error: Optional[str]

    # Downstream workflow outputs
    action: str
    log_entry: str
    explanation_report: str
