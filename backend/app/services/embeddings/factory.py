from backend.app.config import settings
from backend.app.services.embeddings.base import BaseEmbeddingProvider
from backend.app.services.embeddings.mock import MockEmbeddingProvider


def get_embedding_provider(provider_type: str = None, model_name: str = None) -> BaseEmbeddingProvider:
    """
    Factory function returning a concrete BaseEmbeddingProvider instance based on configuration.
    Supports modular swapping of embedding models and providers.
    """
    provider_type = (provider_type or settings.EMBEDDING_PROVIDER).lower()
    model_name = model_name or settings.EMBEDDING_MODEL_NAME

    if provider_type == "mock":
        mock_name = model_name if model_name != settings.EMBEDDING_MODEL_NAME else "mock-minilm-l6-v2"
        return MockEmbeddingProvider(model_name=mock_name, dim=settings.EMBEDDING_DIMENSION)

    if provider_type in {"sentence-transformers", "sentence_transformers"}:
        try:
            from backend.app.services.embeddings.sentence_transformer import SentenceTransformerProvider
            return SentenceTransformerProvider(model_name=model_name)
        except Exception as exc:
            print(f"Warning: Failed to load SentenceTransformers provider ({exc}). Falling back to MockEmbeddingProvider.")
            return MockEmbeddingProvider(model_name=f"{model_name}-fallback", dim=settings.EMBEDDING_DIMENSION)

    raise ValueError(f"Unsupported embedding provider: '{provider_type}'")
