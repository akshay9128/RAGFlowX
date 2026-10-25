from backend.app.config import settings
from backend.app.services.generation.base import BaseLLMProvider
from backend.app.services.generation.failover import FailoverLLMProvider
from backend.app.services.generation.mock import MockLLMProvider


def get_llm_provider(provider_type: str = None, model_name: str = None) -> BaseLLMProvider:
    """
    Factory function returning a concrete BaseLLMProvider instance based on configuration.
    Supports modular swapping of LLM models and providers.
    """
    provider_type = (provider_type or settings.LLM_PROVIDER).lower()
    model_name = model_name or settings.LLM_MODEL_NAME

    if settings.LLM_FAILOVER_ENABLED and provider_type != "mock":
        return FailoverLLMProvider()

    if provider_type == "mock":
        return MockLLMProvider(model_name="mock-grounded-llm")

    if provider_type == "gemini":
        try:
            from backend.app.services.generation.gemini import GeminiLLMProvider
            return GeminiLLMProvider(model_name=model_name)
        except Exception as exc:
            print(f"Warning: Failed to load Gemini LLM provider ({exc}). Falling back to MockLLMProvider.")
            return MockLLMProvider(model_name=f"{model_name}-fallback")

    if provider_type == "openai":
        try:
            from backend.app.services.generation.openai import OpenAILLMProvider
            return OpenAILLMProvider(model_name=model_name)
        except Exception as exc:
            print(f"Warning: Failed to load OpenAI LLM provider ({exc}). Falling back to MockLLMProvider.")
            return MockLLMProvider(model_name=f"{model_name}-fallback")

    raise ValueError(f"Unsupported LLM provider: '{provider_type}'")
