from typing import Optional
from fastapi import APIRouter, HTTPException, status, Query
from backend.app.schemas.chunk import (
    ChunkingRequest,
    ChunkResponse,
    DocumentChunksResponse
)
from backend.app.models.chunk import ChunkingConfig
from backend.app.services.chunking.service import chunking_service
from backend.app.database.chunk_store import chunk_store

router = APIRouter(prefix="/chunks", tags=["Chunks"])


@router.post(
    "/process/{document_id}",
    response_model=DocumentChunksResponse,
    status_code=status.HTTP_200_OK,
    summary="Process document text into chunks with metadata"
)
async def process_chunks(
    document_id: str,
    chunk_size: Optional[int] = Query(default=500, gt=0, le=10000, description="Target chunk character size"),
    chunk_overlap: Optional[int] = Query(default=50, ge=0, description="Character overlap between chunks")
):
    """Splits an ingested document into text chunks while preserving document and page metadata."""
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_overlap must be strictly less than chunk_size."
        )

    config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunking_service.process_document_chunks(document_id, config)

    chunk_responses = [
        ChunkResponse(
            chunk_id=c.metadata.chunk_id,
            document_id=c.metadata.document_id,
            filename=c.metadata.filename,
            page_number=c.metadata.page_number,
            chunk_index=c.metadata.chunk_index,
            text=c.text,
            character_count=c.metadata.character_count,
            source=c.metadata.source
        )
        for c in chunks
    ]

    return DocumentChunksResponse(
        document_id=document_id,
        total_chunks=len(chunk_responses),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunks=chunk_responses
    )


@router.get(
    "/{document_id}",
    response_model=DocumentChunksResponse,
    summary="Get stored chunks for a document"
)
async def get_chunks(document_id: str):
    """Retrieves all generated chunks and metadata for a specific document ID."""
    chunks = chunk_store.get_chunks_by_document(document_id)
    if not chunks:
        # Check if document exists
        from backend.app.database.document_store import document_store
        doc = document_store.get_document(document_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID '{document_id}' not found."
            )

    chunk_responses = [
        ChunkResponse(
            chunk_id=c.metadata.chunk_id,
            document_id=c.metadata.document_id,
            filename=c.metadata.filename,
            page_number=c.metadata.page_number,
            chunk_index=c.metadata.chunk_index,
            text=c.text,
            character_count=c.metadata.character_count,
            source=c.metadata.source
        )
        for c in chunks
    ]

    return DocumentChunksResponse(
        document_id=document_id,
        total_chunks=len(chunk_responses),
        chunk_size=500,
        chunk_overlap=50,
        chunks=chunk_responses
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete chunks for a document"
)
async def delete_chunks(document_id: str):
    """Deletes all generated chunks for a document ID."""
    deleted = chunk_store.delete_chunks_by_document(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No chunks found for document ID '{document_id}'."
        )
    return {"message": f"Chunks for document '{document_id}' deleted successfully."}
