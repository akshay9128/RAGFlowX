import pytest
import inspect
from typing import Optional
from backend.app.services.generation.base import BaseLLMProvider
from backend.app.services.generation.gemini import GeminiLLMProvider
from backend.app.services.generation.openai import OpenAILLMProvider
from backend.app.services.generation.mock import MockLLMProvider
from backend.app.services.generation.failover import FailoverLLMProvider
from backend.app.services.retrieval.rag_service import RAGService


class RecordingStubProvider(BaseLLMProvider):
    """Stub provider that records calls to generate() for testing argument forwarding."""
    def __init__(self, name: str = "recording-stub", fail: bool = False):
        self._name = name
        self.fail = fail
        self.calls = []

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return "recording-model"

    def generate(self, prompt: str, system_instruction: Optional[str] = None,
                 use_chain_of_thought: bool = False) -> str:
        self.calls.append({
            "prompt": prompt,
            "system_instruction": system_instruction,
            "use_chain_of_thought": use_chain_of_thought
        })
        if self.fail:
            raise RuntimeError(f"Provider {self._name} simulated failure")
        return f"Response from {self._name} (CoT={use_chain_of_thought})"


@pytest.mark.parametrize("provider_cls,kwargs", [
    (GeminiLLMProvider, {"model_name": "test", "api_key": "dummy"}),
    (OpenAILLMProvider, {"model_name": "test", "api_key": "dummy"}),
    (MockLLMProvider, {"model_name": "test"}),
    (FailoverLLMProvider, {"provider_order": ["mock"]}),
])
def test_individual_provider_signature_cot(provider_cls, kwargs):
    """Verify each provider class generate() accepts use_chain_of_thought with default False."""
    p = provider_cls(**kwargs)
    sig = inspect.signature(p.generate)
    assert "use_chain_of_thought" in sig.parameters, (
        f"Provider {p.provider_name}.generate() missing 'use_chain_of_thought' parameter"
    )
    param = sig.parameters["use_chain_of_thought"]
    assert param.default is False, (
        f"Provider {p.provider_name}.generate() 'use_chain_of_thought' default must be False, got {param.default}"
    )


def test_failover_forwards_use_chain_of_thought():
    """Verify FailoverLLMProvider correctly forwards use_chain_of_thought to the underlying provider."""
    stub1 = RecordingStubProvider(name="stub1", fail=True)
    stub2 = RecordingStubProvider(name="stub2", fail=False)

    # Monkeypatch FailoverLLMProvider._instantiate to return our recording stubs in order
    failover = FailoverLLMProvider(provider_order=["stub1", "stub2"])
    failover._instantiate = lambda t: {"stub1": stub1, "stub2": stub2}[t]

    res = failover.generate("test prompt", system_instruction="sys", use_chain_of_thought=True)

    assert "stub2" in res
    assert len(stub1.calls) == 1
    assert stub1.calls[0]["use_chain_of_thought"] is True
    assert len(stub2.calls) == 1
    assert stub2.calls[0]["use_chain_of_thought"] is True


def test_rag_service_with_incompatible_provider_fails(monkeypatch):
    """Verify RAGService query raises TypeError when LLM provider lacks use_chain_of_thought."""
    class IncompatibleStub(BaseLLMProvider):
        @property
        def provider_name(self): return "incompatible"
        @property
        def model_name(self): return "inc"
        def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
            return "ok"

    rag = RAGService(llm_provider=IncompatibleStub())
    from backend.app.models.search import VectorSearchResult
    monkeypatch.setattr(
        rag.vec_service,
        "search_similar_chunks",
        lambda query, top_k=5, document_id=None: [
            VectorSearchResult(chunk_id="1", document_id="doc1", filename="test.pdf", page_number=1, chunk_index=0, text="sample text", score=0.9, source="test.pdf")
        ]
    )
    with pytest.raises(TypeError):
        rag.query_rag("test question", use_chain_of_thought=True)
