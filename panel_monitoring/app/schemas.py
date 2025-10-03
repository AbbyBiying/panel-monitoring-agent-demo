# app/schemas.py
from typing import Literal, TypedDict, Dict, Any
from pydantic import BaseModel, Field


# Structured schema for LLM output
class Signals(BaseModel):
    suspicious_signup: bool = Field(..., description="True if the event is suspicious")
    normal_signup: bool = Field(..., description="True if the event is normal")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str


class GraphState(TypedDict):
    event_data: str
    signals: Dict[str, Any]
    classification: Literal["suspicious", "normal", "error"]
    action: str
    log_entry: str
    explanation_report: str
