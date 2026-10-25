from backend.app.config import settings
from backend.app.services.reranking.base import BaseRerankProvider
from backend.app.services.reranking.mock import MockReranker


def get_rerank_provider(provider_type: str = None, model_name: str = None) -> BaseRerankProvider:
    """Factory function returning a concrete BaseRerankProvider instance."""
    provider_type = (provider_type or getattr(settings, "RERANK_PROVIDER", "sentence-transformers")).lower()
    model_name = model_name or getattr(settings, "RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    if provider_type == "mock":
        return MockReranker(model_name=model_name if model_name != "cross-encoder/ms-marco-MiniLM-L-6-v2" else "mock-cross-encoder")

    if provider_type in {"sentence-transformers", "sentence_transformers"}:
        try:
            from backend.app.services.reranking.sentence_transformer import SentenceTransformerReranker
            return SentenceTransformerReranker(model_name=model_name)
        except Exception as exc:
            print(f"Warning: Failed to load SentenceTransformerReranker ({exc}). Falling back to MockReranker.")
            return MockReranker(model_name="mock-cross-encoder-fallback")

    raise ValueError(f"Unsupported rerank provider: '{provider_type}'")
