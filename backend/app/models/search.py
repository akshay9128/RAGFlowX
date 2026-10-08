from typing import Optional
from pydantic import BaseModel, Field
from backend.app.models.chunk import ChunkMetadata


class VectorSearchResult(BaseModel):
    """Represents a single retrieved chunk with its similarity score."""
    chunk_id: str = Field(..., description="Unique ID of the chunk")
    document_id: str = Field(..., description="Parent document ID")
    filename: str = Field(..., description="Parent document filename")
    page_number: int = Field(..., description="1-indexed page number")
    chunk_index: int = Field(..., description="Sequence index of chunk in document")
    text: str = Field(..., description="Chunk text content")
    score: float = Field(..., description="Similarity score (e.g. Cosine Similarity between 0.0 and 1.0)")
    source: str = Field(..., description="File path reference")
