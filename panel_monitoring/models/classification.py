# panel_monitoring/models/classification.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any

from panel_monitoring.models.base import BaseModel


@dataclass
class Classification(BaseModel):
    """
    Postgres table: classification
    Inherits id, uuid, created_at, modified_at from BaseModel
    """

    model_name: str = ""
    classification_duration: float = 0.0
    signals: Dict[str, Any] = field(default_factory=dict)
    classification: str = ""  # e.g. "normal" or "suspicious"
    action: str = ""  # e.g. "no_action", "alert"
