from typing import List, Optional, Tuple
from sentence_transformers import CrossEncoder
from backend.app.services.reranking.base import BaseRerankProvider


class SentenceTransformerReranker(BaseRerankProvider):
    """Cross-encoder reranker using sentence_transformers CrossEncoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = CrossEncoder(model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        scores = self._model.predict(pairs)

        scored = [(idx, float(score)) for idx, score in enumerate(scores)]
        scored.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        return scored
