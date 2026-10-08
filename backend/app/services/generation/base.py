from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMProvider(ABC):
    """Abstract Base Class for modular LLM generation providers (Gemini, OpenAI, Mock, Ollama)."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns provider identifier name."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns LLM model name."""
        pass

    @abstractmethod
    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generates a text completion response given prompt and optional system instructions."""
        pass
