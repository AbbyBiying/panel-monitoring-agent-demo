# panel_monitoring/app/clients/llms/__init__.py

from panel_monitoring.app.prompts import PROMPT_CLASSIFY_SYSTEM, PROMPT_CLASSIFY_USER
from .openai import LLMClientOpenAI, classify_with_openai
from .gemini import LLMClientGemini, classify_with_genai
from .vertexai import LLMClientVertexAI, classify_with_vertexai
# Factory for CLI or programmatic use
def get_llm_classifier(provider: str):
    """
    Returns a callable classification function for the chosen provider.
    Example:
        classify = get_llm_classifier("openai")
        result = classify("some event text")
    """
    provider = provider.lower()
    if provider == "openai":
        return classify_with_openai
    elif provider in ("genai", "gemini"):
        return classify_with_genai
    elif provider == "vertexai":
        return classify_with_vertexai
    else:
        raise ValueError(f"Unknown provider: {provider}")
