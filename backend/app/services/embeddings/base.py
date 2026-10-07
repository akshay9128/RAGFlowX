from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """Abstract Base Class for modular embedding providers (SentenceTransformers, OpenAI, Gemini, etc.)."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the identifier name of the embedding model."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the vector dimension produced by this embedding model."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generates a dense vector embedding for a single text query string."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates dense vector embeddings for a list of document text strings."""
        pass
