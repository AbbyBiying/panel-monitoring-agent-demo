# panel_monitoring/models/base.py
from __future__ import annotations
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class BaseModel:
    """
    Minimal base model following Postgres conventions:
    - id: int (auto-incremented by DB, starts as None in Python)
    - uuid: uuid4, unique, indexed
    - created_at / modified_at: timestamps
    """

    id: Optional[int] = None  # filled by Postgres when inserted
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)

    def touch(self) -> None:
        """Update modified_at before saving."""
        self.modified_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, ISO formatting for datetimes."""
        data = asdict(self)
        for k, v in data.items():
            if isinstance(v, datetime):
                data[k] = v.isoformat()
        return data
