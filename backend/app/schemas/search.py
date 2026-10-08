from typing import List, Optional
from pydantic import BaseModel, Field


class VectorSearchRequest(BaseModel):
    """API request schema for vector similarity search."""
    query: str = Field(..., description="User question or query string", min_length=1)
    top_k: Optional[int] = Field(default=5, description="Number of top relevant chunks to retrieve", gt=0, le=100)
    document_id: Optional[str] = Field(default=None, description="Optional document ID to filter search scope")


class VectorSearchResultItem(BaseModel):
    """Schema for individual retrieved search result item."""
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    score: float
    source: str


class VectorSearchResponse(BaseModel):
    """API response schema for vector similarity search."""
    query: str
    total_results: int
    results: List[VectorSearchResultItem]


class VectorStoreStatsResponse(BaseModel):
    """API response schema for vector store statistics."""
    provider: str
    total_vectors: int
    dimension: int
