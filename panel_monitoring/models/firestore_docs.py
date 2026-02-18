# panel_monitoring/models/firestore_docs.py
from __future__ import annotations
import enum
from datetime import UTC, datetime
from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field, field_validator


class PromptDeploymentStatus(enum.StrEnum):
    DEACTIVATED = enum.auto()
    PRE_LIVE = enum.auto()
    CANARY = enum.auto()
    LIVE = enum.auto()
    FAILOVER = enum.auto()


class PromptModelHost(enum.StrEnum):
    VERTEXAI = enum.auto()
    GEMINI = enum.auto()
    OPENAI = enum.auto()
    ANTHROPIC = enum.auto()


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
    panelist_id: Optional[str] = None
    prompt_id: Optional[str] = None
    prompt_name: Optional[str] = None


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


class PromptSpecDoc(_BaseDoc):
    """Firestore document for a prompt specification (mirrors app_portal PromptSpec)."""

    model_host: PromptModelHost = PromptModelHost.VERTEXAI
    model_name: str = ""
    system_prompt: str = ""
    user_prompt: str = ""
    config: Dict[str, Any] = Field(default_factory=dict)
    version: str = ""
    labels: List[str] = Field(default_factory=list)
    url: Optional[str] = None

    deployment_status: PromptDeploymentStatus = PromptDeploymentStatus.DEACTIVATED
    deployment_role: Optional[str] = None  # e.g. "signup_classification"

    # Not stored in Firestore — populated at read time with the document ID
    doc_id: Optional[str] = Field(default=None, exclude=True)


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
