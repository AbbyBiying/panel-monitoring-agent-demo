# panel_monitoring/app/clients/llms/__init__.py

import os
from typing import Optional

from .openai import LLMClientOpenAI
from .gemini import LLMClientGemini
from .vertexai import LLMClientVertexAI

from .openai import classify_with_openai
from .gemini import classify_with_genai
from .vertexai import classify_with_vertexai


# Pre-initialized client singleton (initialized at startup, before async)
_initialized_client: Optional[object] = None
_initialized_provider: Optional[str] = None


def init_llm_client(provider: Optional[str] = None) -> None:
    """
    Initialize the LLM client at startup (before async event loop).
    Call this from build_graph() or application startup.

    This performs all blocking I/O (credential loading, etc.) synchronously
    at startup time, so async nodes can use the client without blocking.
    """
    global _initialized_client, _initialized_provider

    if _initialized_client is not None:
        return  # Already initialized

    provider = provider or os.getenv("PANEL_DEFAULT_PROVIDER", "vertexai")
    provider = provider.lower()

    if provider == "vertexai":
        client = LLMClientVertexAI()
    elif provider in ("genai", "gemini"):
        client = LLMClientGemini()
    elif provider == "openai":
        client = LLMClientOpenAI()
    else:
        raise ValueError(f"Unknown provider '{provider}'. Valid: vertexai, genai, gemini, openai")

    # This is the blocking call - happens at startup before async
    client.setup()

    _initialized_client = client
    _initialized_provider = provider


def get_initialized_client():
    """
    Get the pre-initialized LLM client.
    Raises if init_llm_client() was not called first.
    """
    if _initialized_client is None:
        raise RuntimeError(
            "LLM client not initialized. Call init_llm_client() at startup."
        )
    return _initialized_client


async def aclassify_event(event: str) -> dict:
    """
    Async classification using the pre-initialized client.
    No blocking I/O - client was initialized at startup.
    """
    client = get_initialized_client()
    return await client.aclassify_event(event)


_CLASSIFIERS = {
    "openai": classify_with_openai,
    "genai": classify_with_genai,
    "gemini": classify_with_genai,
    "vertexai": classify_with_vertexai,
}


def get_llm_classifier(provider: str):
    """
    Return a callable classification function for the chosen provider.
    For sync use cases only.
    """
    key = provider.lower()
    if key not in _CLASSIFIERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Valid options: {', '.join(_CLASSIFIERS.keys())}"
        )
    return _CLASSIFIERS[key]
