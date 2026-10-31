import logging
from typing import List, Optional
from backend.app.config import settings
from backend.app.services.generation.base import BaseLLMProvider
from backend.app.services.generation.gemini import GeminiLLMProvider
from backend.app.services.generation.mock import MockLLMProvider
from backend.app.services.generation.openai import OpenAILLMProvider


logger = logging.getLogger("ragflowx.generation.failover")


class FailoverLLMProvider(BaseLLMProvider):
    """LLM provider that tries providers in order and falls back on failure.

    Ordered list: Gemini -> OpenAI -> Mock (configurable via LLM_FAILOVER_PROVIDERS).
    The last provider in the list is expected to be the Mock provider, which never fails.
    """

    def __init__(self, provider_order: Optional[List[str]] = None):
        if provider_order is None:
            provider_order = [
                p.strip().lower() for p in settings.LLM_FAILOVER_PROVIDERS.split(",")
                if p.strip()
            ]
        if not provider_order:
            provider_order = ["mock"]
        self.provider_order = provider_order

    @property
    def provider_name(self) -> str:
        return "failover"

    @property
    def model_name(self) -> str:
        return ",".join(self.provider_order)

    def _instantiate(self, provider_type: str):
        if provider_type == "gemini":
            model_name = settings.LLM_MODEL_NAME
            return GeminiLLMProvider(model_name=model_name)
        elif provider_type == "openai":
            model_name = settings.LLM_MODEL_NAME
            return OpenAILLMProvider(model_name=model_name)
        elif provider_type == "mock":
            return MockLLMProvider(model_name="mock-grounded-llm")
        raise ValueError(f"Unsupported failover provider: '{provider_type}'")

    def generate(self, prompt: str, system_instruction: Optional[str] = None, use_chain_of_thought: bool = False) -> str:
        last_error: Optional[str] = None
        for provider_type in self.provider_order:
            try:
                provider = self._instantiate(provider_type)
                result = provider.generate(prompt, system_instruction, use_chain_of_thought=use_chain_of_thought)
                logger.info(
                    "FailoverLLMProvider succeeded with provider '%s'", provider_type
                )
                return result
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "FailoverLLMProvider provider '%s' failed: %s", provider_type, exc
                )
                continue

        raise ValueError(
            f"All failover providers failed. Last error: {last_error}"
        )
