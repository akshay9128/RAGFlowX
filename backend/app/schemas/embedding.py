from typing import List, Optional
from pydantic import BaseModel, Field


class TextEmbeddingRequest(BaseModel):
    """Request payload to generate a dense embedding vector for arbitrary text."""
    text: str = Field(..., description="Input text string to embed", min_length=1)


class TextEmbeddingResponse(BaseModel):
    """Response payload containing generated dense vector embedding."""
    text: str
    dimension: int
    embedding_model: str
    embedding: List[float]


class ChunkEmbeddingItem(BaseModel):
    """Schema for individual chunk embedding item."""
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    dimension: int
    embedding_model: str
    embedding: List[float]


class DocumentEmbeddingsResponse(BaseModel):
    """Response payload containing generated embeddings for all document chunks."""
    document_id: str
    total_embeddings: int
    dimension: int
    embedding_model: str
    embeddings: List[ChunkEmbeddingItem]
