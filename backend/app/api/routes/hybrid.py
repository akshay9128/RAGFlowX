from typing import Optional
from fastapi import APIRouter
from backend.app.schemas.search import (
    HybridSearchRequest,
    HybridSearchResponse,
    HybridSearchResultItem
)
from backend.app.services.retrieval.hybrid_service import hybrid_search_service

router = APIRouter(prefix="/hybrid", tags=["Hybrid Search"])


@router.post(
    "/search",
    response_model=HybridSearchResponse,
    summary="Perform hybrid search (Vector + BM25)"
)
async def search_hybrid(payload: HybridSearchRequest):
    """
    Performs hybrid search combining dense vector similarity search and BM25 keyword search
    using Reciprocal Rank Fusion (RRF). Returns the best results from both methods.
    """
    results = hybrid_search_service.search(
        query=payload.query,
        top_k=payload.top_k,
        document_id=payload.document_id
    )

    items = [
        HybridSearchResultItem(
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            filename=r.filename,
            page_number=r.page_number,
            chunk_index=r.chunk_index,
            text=r.text,
            score=r.score,
            source=r.source,
            metadata=r.metadata if hasattr(r, 'metadata') else None
        )
        for r in results
    ]

    return HybridSearchResponse(
        query=payload.query,
        total_results=len(items),
        results=items
    )


@router.get(
    "/stats",
    summary="Get hybrid search statistics"
)
async def get_hybrid_stats():
    """Returns statistics for hybrid search including vector and BM25 components."""
    stats = hybrid_search_service.get_stats()
    return stats