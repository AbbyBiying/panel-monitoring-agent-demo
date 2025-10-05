# panel_monitoring/app/clients/llms/provider_base.py

"""
Back-compat shim so legacy imports keep working.
Providers can be used as callables or objects with .classify(...).
Now they return both structured signals and model metadata.
"""

from typing import Callable, TypedDict, Protocol, Tuple, Dict, Any


class SignalsDict(TypedDict, total=False):
    suspicious_signup: bool
    normal_signup: bool
    confidence: float
    reason: str


class MetaDict(TypedDict, total=False):
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


class ClassifierProvider(Protocol):
    def __call__(self, event: str) -> Tuple[SignalsDict, MetaDict]: ...
    def classify(self, event: str) -> Tuple[SignalsDict, MetaDict]: ...


class FunctionProvider:
    """
    Wrap a function(event: str) -> Tuple[SignalsDict, MetaDict]
    as a provider object.
    """

    def __init__(self, fn: Callable[[str], Tuple[SignalsDict, MetaDict]]) -> None:
        self._fn = fn

    def __call__(self, event: str) -> Tuple[SignalsDict, MetaDict]:
        return self._fn(event)

    def classify(self, event: str) -> Tuple[SignalsDict, MetaDict]:
        return self._fn(event)
