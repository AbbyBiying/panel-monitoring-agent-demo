# Back-compat shim so legacy imports keep working.
# Allows code to treat providers as either callables or objects with .classify(...).

from typing import Callable, TypedDict, Protocol


class SignalsDict(TypedDict):
    suspicious_signup: bool
    normal_signup: bool
    confidence: float
    reason: str


class ClassifierProvider(Protocol):
    def __call__(self, event: str) -> SignalsDict: ...
    def classify(self, event: str) -> SignalsDict: ...


class FunctionProvider:
    """Wrap a function(event: str) -> SignalsDict as a provider object."""

    def __init__(self, fn: Callable[[str], SignalsDict]) -> None:
        self._fn = fn

    def __call__(self, event: str) -> SignalsDict:
        return self._fn(event)

    def classify(self, event: str) -> SignalsDict:
        return self._fn(event)
