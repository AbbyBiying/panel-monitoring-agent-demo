# panel_monitoring/app/clients/llms/base.py

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union


class PredictionError(Exception):
    def __init__(self, message: str, model_ref: str, duration_ms: float = 0.0):
        super().__init__(message)
        self.message = message
        self.model_ref = model_ref
        self.duration_ms = duration_ms


@dataclass(frozen=True)
class PredictionResult:
    model_ref: str
    prompt: str
    text: str
    duration_ms: float


class LLMPredictionClient(ABC):
    def __init__(
        self,
        model_ref: str,
        model_name: str,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        prompt_config: Optional[Union[dict[str, Any], str]] = None,
        log: Any = None,
    ):
        self.model_ref = str(model_ref)
        self.model_name = model_name
        self.system_prompt = system_prompt or None
        self.user_prompt = user_prompt
        if isinstance(prompt_config, str):
            self.prompt_config = json.loads(prompt_config)
        else:
            self.prompt_config = prompt_config or {}
        self.log = log
        self.client = None  # set in setup()

    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    async def predict(self, prompt: str) -> PredictionResult:
        pass
