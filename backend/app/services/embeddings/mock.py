import math
import hashlib
from typing import List
from backend.app.services.embeddings.base import BaseEmbeddingProvider


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic mock embedding provider for fast unit tests or lightweight environments.
    Generates normalized 384-dimensional vectors based on SHA256 string feature hashes.
    """

    def __init__(self, model_name: str = "mock-minilm-l6-v2", dim: int = 384):
        self._model_name = model_name
        self._dimension = dim

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _generate_vector(self, text: str) -> List[float]:
        # Generate pseudo-random deterministic vector from text hash
        vec = []
        for i in range(self._dimension):
            seed_str = f"{text}_{i}"
            hash_val = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
            int_val = int(hash_val[:8], 16)
            # Map to range [-1.0, 1.0]
            val = (int_val / 0xFFFFFFFF) * 2.0 - 1.0
            vec.append(val)

        # L2 Normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_query(self, text: str) -> List[float]:
        return self._generate_vector(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._generate_vector(t) for t in texts]
