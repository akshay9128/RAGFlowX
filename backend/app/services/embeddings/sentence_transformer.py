from typing import List, Optional
from backend.app.services.embeddings.base import BaseEmbeddingProvider

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Embedding provider using SentenceTransformers library."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers library is not installed. "
                "Install it via `pip install sentence-transformers`."
            )
        self._model_name = model_name
        self.model = SentenceTransformer(model_name)
        # Determine vector dimension from sample output
        test_vec = self.model.encode("test", normalize_embeddings=True)
        self._dimension = len(test_vec)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_query(self, text: str) -> List[float]:
        vec = self.model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [[float(x) for x in v] for v in vectors]
