from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.search import (
    BM25SearchRequest,
    BM25SearchResponse,
    BM25StatsResponse,
    VectorSearchResultItem
)
from backend.app.services.retrieval.bm25_service import bm25_search_service

router = APIRouter(prefix="/bm25", tags=["BM25 Keyword Search"])


@router.post(
    "/index/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Build BM25 index for document chunks"
)
async def index_document(document_id: str):
    """Builds BM25 keyword index for all chunks of a document and returns the count indexed."""
    indexed_count = bm25_search_service.index_document(document_id)
    if indexed_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No chunks found for document '{document_id}'. Process the document into chunks first."
        )
    return {
        "document_id": document_id,
        "indexed_chunks": indexed_count,
        "message": f"Successfully indexed {indexed_count} chunks into BM25 index."
    }


@router.post(
    "/search",
    response_model=BM25SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Perform BM25 keyword search"
)
async def search_bm25(payload: BM25SearchRequest):
    """Performs BM25 keyword search and returns top relevant chunks."""
    results = bm25_search_service.search(
        query=payload.query,
        top_k=payload.top_k,
        document_id=payload.document_id
    )

    items = [
        VectorSearchResultItem(
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            filename=r.filename,
            page_number=r.page_number,
            chunk_index=r.chunk_index,
            text=r.text,
            score=r.score,
            source=r.source
        )
        for r in results
    ]

    return BM25SearchResponse(
        query=payload.query,
        total_results=len(items),
        results=items
    )


@router.get(
    "/stats",
    response_model=BM25StatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get BM25 index statistics"
)
async def get_bm25_stats():
    """Returns BM25 index statistics including total indexed chunks and documents."""
    stats = bm25_search_service.get_stats()
    return BM25StatsResponse(
        provider=stats["provider"],
        total_indexed_chunks=stats["total_indexed_chunks"],
        indexed_documents=stats["indexed_documents"]
    )


@router.delete(
    "/document/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete BM25 index for document"
)
async def delete_document_bm25(document_id: str):
    """Deletes BM25 index associated with a document ID."""
    deleted = bm25_search_service.delete_document(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No BM25 index found for document ID '{document_id}'."
        )
    return {"message": f"BM25 index for document '{document_id}' deleted successfully."}