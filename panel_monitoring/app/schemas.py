# panel_monitoring/app/schemas.py
from __future__ import annotations
import operator  # Required for the reducer
from typing import Annotated, Any, Dict, Literal, Optional
from pydantic import BaseModel, Field


# ----------------------------
# Structured schema for LLM output
# ----------------------------


class Signals(BaseModel):
    suspicious_signup: bool = Field(..., description="True if the event is suspicious")
    normal_signup: bool = Field(..., description="True if the event is normal")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0,1]")
    reason: str = Field(
        ..., description="Explanation or rationale for the classification"
    )
    analysis_steps: str = Field(
        ..., 
        description="Telegraphic reasoning. Example: 'Geo: OK. Intent: High. Tech: No flags.'"
    )
    panelist_id: Optional[str] = None


# ----------------------------
# Metadata about model execution
# ----------------------------


class ModelMeta(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[int] = None
    max_output_tokens: Optional[int] = None
    request_timeout: Optional[int] = None
    max_retries: Optional[int] = None
    usage: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: Optional[int] = None
    cost_usd: Optional[float] = None
    error: Optional[str] = None


# ----------------------------
# LangGraph state object
# ----------------------------


def merge_dict(existing: Any, new: Any) -> dict:
    """
    Safe merge for dictionaries.
    Handles cases where 'new' might accidentally be a non-dict.
    """
    if not isinstance(existing, dict):
        existing = {}
    if not isinstance(new, dict):
        # If a node returns a string or None, we ignore it to prevent a crash
        return existing

    return {**existing, **new}


class GraphState(BaseModel):
    # Core identifiers
    project_id: Optional[str] = None
    event_id: Optional[str] = None

    # Input / raw content
    event_text: Optional[str] = None
    # This prevents the "last-node-wins" overwrite bug.
    event_data: Annotated[Dict[str, Any], merge_dict] = Field(default_factory=dict)

    # Classification results
    signals: Optional[Signals] = None
    classification: Literal["pending", "suspicious", "normal", "error"] = "pending"
    confidence: Optional[float] = None
    model_meta: ModelMeta = Field(default_factory=ModelMeta)
    error: Optional[str] = None

    # Downstream workflow outputs
    action: Optional[str] = ""  # default empty to avoid None checks
    log_entry: Optional[str] = None

    # explanation_report: Optional[str] = None
    # We wrap the type in 'Annotated' and provide 'operator.add' as the reducer.
    # This tells LangGraph: "When multiple nodes return this key, add them together without overwriting previous insights."
    explanation_report: Annotated[list[str], operator.add]

    review_decision: Optional[Literal["approve", "reject", "escalate"]] = None
    review_url: Optional[str] = None
    panelist_id: Optional[str] = None
    prompt_id: Optional[str] = None
    prompt_name: Optional[str] = None
    
    # RAG Context: Store the fraud rules retrieved from Firestore
    # We use list[dict] to store both the rule text and its metadata (like rule_id)
    retrieved_docs: Annotated[list[dict], operator.add] = Field(default_factory=list)
