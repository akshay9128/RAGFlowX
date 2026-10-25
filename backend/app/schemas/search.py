from typing import List, Optional
from pydantic import BaseModel, Field


class VectorSearchRequest(BaseModel):
    """Request body for dense vector similarity search."""
    query: str = Field(..., min_length=1, description="Search query string")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of top results to return")
    document_id: Optional[str] = Field(default=None, description="Optional scope to single document")


class VectorSearchResultItem(BaseModel):
    """Individual result item in vector search response."""
    chunk_id: str = Field(..., description="Unique ID of the chunk")
    document_id: str = Field(..., description="Parent document ID")
    filename: str = Field(..., description="Parent document filename")
    page_number: int = Field(..., description="1-indexed page number")
    chunk_index: int = Field(..., description="Sequence index of chunk in document")
    text: str = Field(..., description="Chunk text content")
    score: float = Field(..., description="Similarity score between 0.0 and 1.0")
    source: str = Field(..., description="File path reference")


class VectorSearchResponse(BaseModel):
    """Response body for vector similarity search."""
    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., description="Total number of results returned")
    results: List[VectorSearchResultItem] = Field(..., description="List of search results")


class VectorStoreStatsResponse(BaseModel):
    """Response body for vector store statistics."""
    provider: str = Field(..., description="Vector store provider name")
    total_vectors: int = Field(..., description="Total number of indexed vectors")
    dimension: int = Field(..., description="Embedding vector dimension")


class BM25SearchRequest(BaseModel):
    """Request body for BM25 keyword search."""
    query: str = Field(..., min_length=1, description="Search query string")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of top results to return")
    document_id: Optional[str] = Field(default=None, description="Optional scope to single document")


class BM25SearchResponse(BaseModel):
    """Response body for BM25 keyword search."""
    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., description="Total number of results returned")
    results: List[VectorSearchResultItem] = Field(..., description="List of search results")


class BM25StatsResponse(BaseModel):
    """Response body for BM25 index statistics."""
    provider: str = Field(..., description="BM25 provider name")
    total_indexed_chunks: int = Field(..., description="Total number of indexed chunks")
    indexed_documents: int = Field(..., description="Number of documents with BM25 indices")


class HybridSearchRequest(BaseModel):
    """Request body for hybrid search (vector + BM25)."""
    query: str = Field(..., min_length=1, description="Search query string")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of top results to return")
    document_id: Optional[str] = Field(default=None, description="Optional scope to single document")


class HybridSearchResultItem(BaseModel):
    """Individual result item in hybrid search response."""
    chunk_id: str = Field(..., description="Unique ID of the chunk")
    document_id: str = Field(..., description="Parent document ID")
    filename: str = Field(..., description="Parent document filename")
    page_number: int = Field(..., description="1-indexed page number")
    chunk_index: int = Field(..., description="Sequence index of chunk in document")
    text: str = Field(..., description="Chunk text content")
    score: float = Field(..., description="RRF combined score")
    source: str = Field(..., description="File path reference")
    metadata: Optional[dict] = Field(default=None, description="Additional scoring metadata")


class HybridSearchResponse(BaseModel):
    """Response body for hybrid search."""
    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., description="Total number of results returned")
    results: List[HybridSearchResultItem] = Field(..., description="List of hybrid search results")