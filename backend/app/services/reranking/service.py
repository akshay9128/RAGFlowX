from typing import List, Optional
from backend.app.models.search import SearchResult
from backend.app.services.reranking.base import BaseRerankProvider
from backend.app.services.reranking.factory import get_rerank_provider


class RerankService:
    """Service orchestrating cross-encoder reranking of retrieved search results."""

    def __init__(self, provider: Optional[BaseRerankProvider] = None):
        self._provider = provider

    @property
    def provider(self) -> BaseRerankProvider:
        if self._provider is None:
            self._provider = get_rerank_provider()
        return self._provider

    def set_provider(self, provider: BaseRerankProvider):
        self._provider = provider

    def rerank_results(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Reranks a list of SearchResult objects using the cross-encoder provider.
        """
        if not results:
            return []

        document_texts = [r.text for r in results]
        reranked_pairs = self.provider.rerank(query, document_texts, top_k=top_k)

        reranked_results: List[SearchResult] = []
        for rank, (orig_idx, score) in enumerate(reranked_pairs, start=1):
            if orig_idx < len(results):
                original = results[orig_idx]
                reranked_results.append(
                    SearchResult(
                        chunk_id=original.chunk_id,
                        document_id=original.document_id,
                        filename=original.filename,
                        page_number=original.page_number,
                        chunk_index=original.chunk_index,
                        text=original.text,
                        score=score,
                        source=original.source
                    )
                )

        return reranked_results


rerank_service = RerankService()
