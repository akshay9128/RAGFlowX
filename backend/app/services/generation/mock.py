import re
from typing import Optional
from backend.app.services.generation.base import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """
    Deterministic grounded Mock LLM generator for testing and offline environments.
    Extracts relevant facts directly from the prompt context to synthesize a grounded answer.
    """

    def __init__(self, model_name: str = "mock-grounded-llm"):
        self._model_name = model_name

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Synthesizes a grounded answer based strictly on context blocks in prompt."""
        if not prompt or "No relevant document chunks found" in prompt:
            return "I could not find enough relevant information in the uploaded documents to answer your question."

        # Extract Context block from prompt
        context_match = re.search(r"--- CONTEXT START ---\n(.*?)\n--- CONTEXT END ---", prompt, re.DOTALL)
        question_match = re.search(r"USER QUESTION:\s*(.*?)$", prompt, re.DOTALL)

        if context_match:
            context_text = context_match.group(1).strip()
            question_text = question_match.group(1).strip() if question_match else "your question"

            # Clean context snippets
            clean_snippets = []
            for line in context_text.split("\n"):
                if line.startswith("[Source") or not line.strip():
                    continue
                clean_snippets.append(line.strip())

            combined_facts = " ".join(clean_snippets)
            if combined_facts:
                return f"Based on the provided documents: {combined_facts}"

        return "I could not find enough relevant information in the uploaded documents to answer your question."
