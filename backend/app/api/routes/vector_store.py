from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.search import (
    VectorSearchRequest,
    VectorSearchResponse,
    VectorSearchResultItem,
    VectorStoreStatsResponse
)
from backend.app.services.retrieval.vector_service import vector_search_service

router = APIRouter(prefix="/vector-store", tags=["Vector Store"])


@router.post(
    "/index/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Generate embeddings and index document chunks into vector database"
)
async def index_document(document_id: str):
    """Generates dense embeddings for all chunks of a document and indexes them into the vector store."""
    indexed_count = vector_search_service.index_document_chunks(document_id)
    return {
        "document_id": document_id,
        "indexed_chunks": indexed_count,
        "message": f"Successfully indexed {indexed_count} chunks into vector database."
    }


@router.post(
    "/search",
    response_model=VectorSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Perform dense vector similarity search"
)
async def search_vectors(payload: VectorSearchRequest):
    """Converts user query into dense vector, searches vector database, and returns top relevant chunks."""
    results = vector_search_service.search_similar_chunks(
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

    return VectorSearchResponse(
        query=payload.query,
        total_results=len(items),
        results=items
    )


@router.get(
    "/stats",
    response_model=VectorStoreStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get vector store statistics"
)
async def get_vector_store_stats():
    """Returns vector database stats including provider name, total indexed vectors, and dimension."""
    stats = vector_search_service.get_stats()
    return VectorStoreStatsResponse(
        provider=stats["provider"],
        total_vectors=stats["total_vectors"],
        dimension=stats["dimension"]
    )


@router.delete(
    "/document/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete document vectors"
)
async def delete_document_vectors(document_id: str):
    """Deletes all indexed vectors associated with a document ID."""
    deleted = vector_search_service.delete_document_vectors(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No vectors found for document ID '{document_id}'."
        )
    return {"message": f"Vectors for document '{document_id}' deleted successfully."}
