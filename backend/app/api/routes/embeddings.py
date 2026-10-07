from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.embedding import (
    TextEmbeddingRequest,
    TextEmbeddingResponse,
    DocumentEmbeddingsResponse,
    ChunkEmbeddingItem
)
from backend.app.services.embeddings.service import embedding_service
from backend.app.config import settings

router = APIRouter(prefix="/embeddings", tags=["Embeddings"])


@router.post(
    "/text",
    response_model=TextEmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate dense vector embedding for input text"
)
async def embed_text(payload: TextEmbeddingRequest):
    """Generates a dense vector embedding for any input string."""
    vector = embedding_service.embed_text(payload.text)
    provider = embedding_service.provider

    return TextEmbeddingResponse(
        text=payload.text,
        dimension=provider.dimension,
        embedding_model=provider.model_name,
        embedding=vector
    )


@router.post(
    "/chunks/{document_id}",
    response_model=DocumentEmbeddingsResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate dense embeddings for all chunks of a document"
)
async def embed_document_chunks(document_id: str):
    """Generates dense vector embeddings for every chunk of an ingested document."""
    chunk_embeddings = embedding_service.embed_document_chunks(document_id)
    provider = embedding_service.provider

    items = [
        ChunkEmbeddingItem(
            chunk_id=item.chunk.metadata.chunk_id,
            document_id=item.chunk.metadata.document_id,
            filename=item.chunk.metadata.filename,
            page_number=item.chunk.metadata.page_number,
            chunk_index=item.chunk.metadata.chunk_index,
            dimension=item.dimension,
            embedding_model=item.embedding_model,
            embedding=item.embedding
        )
        for item in chunk_embeddings
    ]

    return DocumentEmbeddingsResponse(
        document_id=document_id,
        total_embeddings=len(items),
        dimension=provider.dimension,
        embedding_model=provider.model_name,
        embeddings=items
    )


@router.get(
    "/info",
    status_code=status.HTTP_200_OK,
    summary="Get active embedding provider configuration details"
)
async def get_embedding_info():
    """Returns the current embedding provider, model name, and vector dimensionality."""
    provider = embedding_service.provider
    return {
        "provider": settings.EMBEDDING_PROVIDER,
        "model_name": provider.model_name,
        "dimension": provider.dimension
    }
