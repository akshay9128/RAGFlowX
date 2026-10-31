import httpx
from typing import Optional
from backend.app.config import settings
from backend.app.services.generation.base import BaseLLMProvider


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI LLM provider using the Chat Completions REST API."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self._model_name = model_name or settings.LLM_MODEL_NAME or "gpt-4o-mini"
        self._api_key = api_key or settings.OPENAI_API_KEY or None

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, system_instruction: Optional[str] = None, use_chain_of_thought: bool = False) -> str:
        """Generates a text completion response via OpenAI Chat Completions API."""
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not configured.")

        effective_system = system_instruction
        if use_chain_of_thought:
            cot_prefix = "Think carefully and explain your reasoning step by step before arriving at the final grounded answer."
            effective_system = f"{system_instruction}\n{cot_prefix}" if system_instruction else cot_prefix

        messages = []
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": 0.2,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                )
                if res.status_code != 200:
                    raise ValueError(
                        f"OpenAI API returned status code {res.status_code}: {res.text}"
                    )
                data = res.json()
                choices = data.get("choices", [])
                if choices and "message" in choices[0]:
                    return choices[0]["message"].get("content", "").strip()
                return "OpenAI API returned an empty response."
        except Exception as exc:
            raise ValueError(f"Failed to generate text from OpenAI API: {str(exc)}")
