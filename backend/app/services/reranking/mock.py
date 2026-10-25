from typing import List, Optional, Tuple
from backend.app.services.reranking.base import BaseRerankProvider


class MockReranker(BaseRerankProvider):
    """Deterministic mock reranker for offline testing and development."""

    def __init__(self, model_name: str = "mock-cross-encoder"):
        self._model_name = model_name

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

        query_tokens = set(query.lower().split())
        scored = []

        for idx, doc in enumerate(documents):
            doc_tokens = set(doc.lower().split())
            overlap = len(query_tokens & doc_tokens)
            # Add a small deterministic score based on position to avoid ties
            score = float(overlap) + (1.0 / (idx + 1.0) * 0.1)
            scored.append((idx, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        return scored
