import os
import httpx
from typing import Optional
from backend.app.config import settings
from backend.app.services.generation.base import BaseLLMProvider


class GeminiLLMProvider(BaseLLMProvider):
    """Google Gemini LLM provider using REST API / google-genai SDK."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self._model_name = model_name or settings.LLM_MODEL_NAME or "gemini-2.5-flash"
        self._api_key = api_key or settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not configured.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model_name}:generateContent?key={self._api_key}"

        contents = []
        if system_instruction:
            contents.append({
                "role": "user",
                "parts": [{"text": f"SYSTEM INSTRUCTION: {system_instruction}\n\n{prompt}"}]
            })
        else:
            contents.append({
                "role": "user",
                "parts": [{"text": prompt}]
            })

        payload = {"contents": contents}

        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(url, json=payload)
                if res.status_code != 200:
                    raise ValueError(f"Gemini API returned status code {res.status_code}: {res.text}")
                data = res.json()
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
                return "Gemini API returned an empty response."
        except Exception as exc:
            raise ValueError(f"Failed to generate text from Gemini API: {str(exc)}")
