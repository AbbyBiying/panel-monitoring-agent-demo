# panel_monitoring/app/clients/llms/provider_base.py

"""
Back-compat shim so legacy imports keep working.
Providers can be used as callables or objects with .classify(...).
Now they return both structured signals and model metadata.
"""

from typing import Callable, TypedDict, Protocol, Tuple, Dict, Any


# UserEvents are recorded and emitted by various platforms the panelist interacts with and are the input to the classifier
class UserEvent(TypedDict, total=False):
    panelist_id: str  # unique alpha-numeric id for the panelist
    type: str  # the type of event (one of event types in the panelist's platform)
    platform_name: str  # the name of the platform the panelist acted on
    observed_at: str  # ISO-8601 timestamp of when the event was observed
    description: str  # a natural language description of the event


# Signals are the output of the classifier and are used to determine if the event is suspicious or normal
class Signal(TypedDict, total=False):
    suspicious_signup: bool
    normal_signup: bool
    confidence: float
    reason: str


# SignalMeta is used to store metadata about the classifier and it's inference
class SignalMeta(TypedDict, total=False):
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
    def __call__(self, event: str) -> Tuple[Signal, SignalMeta]: ...
    def classify(self, event: str) -> Tuple[Signal, SignalMeta]: ...


class FunctionProvider:
    """
    Wrap a function(event: str) -> Tuple[Signal, SignalMeta]
    as a provider object.
    """

    def __init__(self, fn: Callable[[str], Tuple[Signal, SignalMeta]]) -> None:
        self._fn = fn

    def __call__(self, event: str) -> Tuple[Signal, SignalMeta]:
        return self._fn(event)

    def classify(self, event: str) -> Tuple[Signal, SignalMeta]:
        return self._fn(event)
