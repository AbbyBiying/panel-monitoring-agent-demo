# panel_monitoring/models/firestore_docs.py
from __future__ import annotations
from datetime import UTC, datetime
from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field, field_validator


class _BaseDoc(BaseModel):
    """Shared options for Firestore documents."""

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_validator("created_at", mode="before")
    @classmethod
    def _default_created_at(cls, v):
        return v or datetime.now(UTC)

    @field_validator("updated_at", mode="before")
    @classmethod
    def _default_updated_at(cls, v):
        return v or datetime.now(UTC)


class ProjectDoc(_BaseDoc):
    project_id: str
    name: str
    status: Literal["active", "archived"] = "active"
    description: Optional[str] = None
    agents: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EventDoc(_BaseDoc):
    # Required
    project_id: str
    type: str
    source: str
    received_at: datetime
    # Optional
    event_at: Optional[datetime] = None
    user_hash: Optional[str] = None
    ip_hash: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    # classification summary fields
    status: Literal["pending", "classified", "error"] = "pending"
    decision: Optional[str] = None
    confidence: Optional[float] = None
    last_run_id: Optional[str] = None


class RunDoc(_BaseDoc):
    # Required
    run_id: str
    project_id: str
    event_id: str
    agent_id: str
    status: Literal["success", "error", "running", "queued"]

    # Optional
    logs: List[str] = Field(default_factory=list)
    output: Dict[str, Any] = Field(default_factory=dict)

    # Core, queryable fields for analytics / dashboards
    decision: Optional[str] = None
    confidence: Optional[float] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None

    # High-level, safe LLM decision summary (what we built from Signals)
    llm_decision_summary: Optional[Dict[str, Any]] = None


class AlertDoc(_BaseDoc):
    # Required
    alert_id: str
    project_id: str
    event_id: str
    severity: Literal["low", "medium", "high", "critical"]
    message: str
    # Optional
    resolved: bool = False
    context: Dict[str, Any] = Field(default_factory=dict)
