# panel_monitoring/models/prompt_spec.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Dict

from panel_monitoring.models.base import BaseModel


@dataclass
class PromptSpec(BaseModel):
    """
    Postgres table: prompt_spec
    Immutable after creation.
    """

    model_host: str = ""
    model_name: str = ""

    system_prompt: str = ""
    prompt: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    version: str = ""
    labels: List[str] = field(default_factory=list)
