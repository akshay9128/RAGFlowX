from typing import Dict, List, Optional, Tuple
from backend.app.models.search import SearchResult, VectorSearchResult
from backend.app.services.retrieval.vector_service import vector_search_service, VectorSearchService
from backend.app.services.retrieval.bm25_service import bm25_search_service, BM25SearchService


def reciprocal_rank_fusion(
    vector_results: List[SearchResult],
    bm25_results: List[SearchResult],
    top_k: int = 5,
    k: float = 60.0
) -> List[SearchResult]:
    """
    Combines vector search and BM25 search results using Reciprocal Rank Fusion (RRF).

    RRF Score = 1 / (k + rank_in_list)

    Args:
        vector_results: Results from dense vector search
        bm25_results: Results from BM25 keyword search
        top_k: Number of top results to return
        k: RRF constant (default 60, from RRF paper)

    Returns:
        Combined list of SearchResult objects ranked by RRF score
    """
    if not vector_results and not bm25_results:
        return []

    # Create a map of chunk_id -> SearchResult for quick lookup
    result_map: Dict[str, SearchResult] = {}

    # Track ranks for both result lists
    vector_ranks: Dict[str, int] = {}
    bm25_ranks: Dict[str, int] = {}

    # Assign ranks (1-indexed) based on position in each result list
    for rank, result in enumerate(vector_results, start=1):
        vector_ranks[result.chunk_id] = rank

    for rank, result in enumerate(bm25_results, start=1):
        bm25_ranks[result.chunk_id] = rank

    # Collect all unique chunk_ids
    all_chunk_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())

    # Calculate RRF scores for each chunk
    rrf_scores: Dict[str, float] = {}
    for chunk_id in all_chunk_ids:
        vector_rank = vector_ranks.get(chunk_id, float('inf'))
        bm25_rank = bm25_ranks.get(chunk_id, float('inf'))

        rrf_score = 0.0
        if vector_rank != float('inf'):
            rrf_score += 1.0 / (k + vector_rank)
        if bm25_rank != float('inf'):
            rrf_score += 1.0 / (k + bm25_rank)

        rrf_scores[chunk_id] = rrf_score

    # Build combined result list with merged metadata
    for result in vector_results:
        result_map[result.chunk_id] = result

    for result in bm25_results:
        if result.chunk_id not in result_map:
            result_map[result.chunk_id] = result

    # Create combined results with RRF scores
    combined_results: List[Tuple[str, float]] = []
    for chunk_id, rrf_score in rrf_scores.items():
        if chunk_id in result_map:
            combined_results.append((chunk_id, rrf_score))

    # Sort by RRF score descending
    combined_results.sort(key=lambda x: x[1], reverse=True)

    # Build final results list
    final_results: List[SearchResult] = []
    seen_chunks = set()

    for chunk_id, rrf_score in combined_results:
        if chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)

        if chunk_id in result_map:
            original = result_map[chunk_id]
            final_results.append(
                SearchResult(
                    chunk_id=original.chunk_id,
                    document_id=original.document_id,
                    filename=original.filename,
                    page_number=original.page_number,
                    chunk_index=original.chunk_index,
                    text=original.text,
                    score=rrf_score,  # RRF score instead of original score
                    source=original.source,
                    # Optional: include individual scores as metadata
                    metadata={
                        "vector_rank": vector_ranks.get(chunk_id, None),
                        "bm25_rank": bm25_ranks.get(chunk_id, None),
                        "rrf_score": rrf_score
                    }
                )
            )

    return final_results[:top_k]


class HybridSearchService:
    """
    Service orchestrating hybrid search combining dense vector search and BM25 keyword search
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vec_service: VectorSearchService = vector_search_service,
        bm25_service: BM25SearchService = bm25_search_service
    ):
        self.vec_service = vec_service
        self.bm25_service = bm25_service

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Performs hybrid search by combining vector and BM25 results.

        Uses BM25 for authoritative keyword search results.
        Falls back to vector search only when BM25 returns no results
        and a vector store is available with real embeddings.

        Args:
            query: Search query string
            top_k: Number of top results to return
            document_id: Optional scope to single document

        Returns:
            List of SearchResult objects ranked by combined RRF score
        """
        if not query or not query.strip():
            return []

        # Execute BM25 keyword search
        bm25_results = self.bm25_service.search(
            query=query,
            top_k=top_k * 2,
            document_id=document_id
        )

        # Execute dense vector similarity search
        vector_search_results = self.vec_service.search_similar_chunks(
            query=query,
            top_k=top_k * 2,
            document_id=document_id
        )

        # Convert VectorSearchResult to SearchResult format
        vector_results: List[SearchResult] = []
        for r in vector_search_results:
            vector_results.append(
                SearchResult(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    filename=r.filename,
                    page_number=r.page_number,
                    chunk_index=r.chunk_index,
                    text=r.text,
                    score=r.score,
                    source=r.source
                )
            )

        # If both retrieval systems return nothing, there is nothing to fuse
        if not vector_results and not bm25_results:
            return []

        # Combine both ranked lists with Reciprocal Rank Fusion on every query
        return reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            top_k=top_k,
        )

    def get_stats(self) -> dict:
        """Returns statistics for both vector and BM25 search services."""
        vector_stats = {
            "provider": "vector_search",
            "total_vectors": 0,  # Would need to access internal stats
            "dimension": 384
        }
        bm25_stats = self.bm25_service.get_stats()

        return {
            "hybrid": True,
            "vector_search": {
                "enabled": True,
                "provider": "in_memory"
            },
            "bm25_search": bm25_stats,
            "fusion_method": "RRF (Reciprocal Rank Fusion)"
        }


# Global singleton instance
hybrid_search_service = HybridSearchService()