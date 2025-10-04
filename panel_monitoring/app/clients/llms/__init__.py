# panel_monitoring/app/clients/llms/__init__.py

from .openai import classify_with_openai
from .gemini import classify_with_genai
from .vertexai import classify_with_vertexai


_CLASSIFIERS = {
    "openai": classify_with_openai,
    "genai": classify_with_genai,
    "gemini": classify_with_genai,
    "vertexai": classify_with_vertexai,
}

def get_llm_classifier(provider: str):
    """
    Return a callable classification function for the chosen provider.

    Example:
        classify = get_llm_classifier("openai")
        result = classify("some event text")
    """
    key = provider.lower()
    if key not in _CLASSIFIERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Valid options: {', '.join(_CLASSIFIERS.keys())}"
        )
    return _CLASSIFIERS[key]
