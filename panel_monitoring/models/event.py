from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class EventDoc(BaseModel):
    type: str
    source: str
    received_at: datetime
    event_at: Optional[datetime] = None
    user_hash: Optional[str] = None
    ip_hash: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)

    # classification summary fields
    status: str = "pending"  # pending | classified | error
    decision: Optional[str] = None
    confidence: Optional[float] = None
    last_run_id: Optional[str] = None
    updated_at: Optional[datetime] = None

    panelist_id: Optional[str] = None
    # see which prompt this run use
    prompt_name: Optional[str] = None    
    commit_id: Optional[str] = None