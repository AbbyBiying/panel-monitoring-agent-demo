# panel_monitoring/app/clients/llms/__init__.py

import os
import asyncio
from typing import Optional, Dict, Any

from .openai import LLMClientOpenAI
from .gemini import LLMClientGemini
from .vertexai import LLMClientVertexAI

# Pre-initialized client singleton
_initialized_client: Any = None
_initialized_provider: Optional[str] = None
_init_lock = asyncio.Lock() # Guard for async environments

def init_llm_client(provider: Optional[str] = None) -> None:
    """Synchronous initialization for startup/cli."""
    global _initialized_client, _initialized_provider

    if _initialized_client is not None:
        return

    provider = (provider or os.getenv("PANEL_DEFAULT_PROVIDER", "vertexai")).lower()

    clients = {
        "vertexai": LLMClientVertexAI,
        "genai": LLMClientGemini,
        "gemini": LLMClientGemini,
        "openai": LLMClientOpenAI
    }

    if provider not in clients:
        raise ValueError(f"Invalid provider: {provider}. Options: {list(clients.keys())}")

    client = clients[provider]()
    client.setup() # Blocking setup performed at startup

    _initialized_client = client
    _initialized_provider = provider

def get_initialized_client():
    if _initialized_client is None:
        # Fallback for dynamic init if forgot to call at startup
        init_llm_client()
    return _initialized_client


async def aclassify_event(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Production Async Entrypoint."""
    client = get_initialized_client()
    return await client.aclassify_event(event_data)