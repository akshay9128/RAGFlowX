import re
from typing import Optional
from backend.app.services.generation.base import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """
    Deterministic grounded Mock LLM generator for testing and offline environments.
    Extracts relevant facts directly from the prompt context to synthesize a grounded answer.
    Supports optional chain-of-thought reasoning.
    """

    def __init__(self, model_name: str = "mock-grounded-llm"):
        self._model_name = model_name

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, system_instruction: Optional[str] = None,
                 use_chain_of_thought: bool = False) -> str:
        """Synthesizes a grounded answer based strictly on context blocks in prompt."""
        if not prompt or "No relevant document chunks found" in prompt:
            return "I could not find enough relevant information in the uploaded documents to answer your question."

        # Extract Context block from prompt
        context_match = re.search(r"--- CONTEXT START ---\n(.*?)\n--- CONTEXT END ---", prompt, re.DOTALL)
        question_match = re.search(r"USER QUESTION:\s*(.*?)$", prompt, re.DOTALL)

        if context_match:
            context_text = context_match.group(1).strip()
            question_text = question_match.group(1).strip() if question_match else "your question"

            # Build lines of actual content (skip source headers)
            clean_lines = [line.strip() for line in context_text.split("\n")
                           if not line.startswith("[Source") and line.strip()]
            combined_facts = " ".join(clean_lines)

            if combined_facts:
                if use_chain_of_thought:
                    # Chain-of-thought: reason step-by-step over retrieved evidence
                    evidence_summary = combined_facts[:len(combined_facts) // 2]
                    trace = (
                        f"Step 1 - Identify key terms in question: {question_text}\n"
                        f"Step 2 - Match terms against retrieved document context.\n"
                        f"Step 3 - Supporting evidence found: {evidence_summary}\n"
                        f"Step 4 - Compose answer strictly from the evidence above.\n"
                    )
                    return trace + f"ANSWER: Based on the provided documents: {combined_facts}"
                return f"Based on the provided documents: {combined_facts}"

        return "I could not find enough relevant information in the uploaded documents to answer your question."
