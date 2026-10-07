from typing import List, Optional
from pydantic import BaseModel, Field
from backend.app.models.chunk import ChunkMetadata, DocumentChunk


class ChunkingRequest(BaseModel):
    """API request schema for customizing document chunking parameters."""
    chunk_size: Optional[int] = Field(default=500, description="Target character size for chunks", gt=0, le=10000)
    chunk_overlap: Optional[int] = Field(default=50, description="Overlap in characters between consecutive chunks", ge=0)


class ChunkResponse(BaseModel):
    """API schema for returning individual chunk details."""
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    character_count: int
    source: str


class DocumentChunksResponse(BaseModel):
    """API response schema containing all chunks for a document."""
    document_id: str
    total_chunks: int
    chunk_size: int
    chunk_overlap: int
    chunks: List[ChunkResponse]
