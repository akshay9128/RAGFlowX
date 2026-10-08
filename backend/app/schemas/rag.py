from typing import List, Optional
from pydantic import BaseModel, Field
from backend.app.schemas.search import VectorSearchResultItem


class RAGQueryRequest(BaseModel):
    """API request schema for grounded RAG query."""
    query: str = Field(..., description="User question or prompt", min_length=1)
    top_k: Optional[int] = Field(default=5, description="Number of top relevant context chunks to retrieve", gt=0, le=50)
    document_id: Optional[str] = Field(default=None, description="Optional document ID to restrict query scope")


class RAGQueryResponse(BaseModel):
    """API response schema for grounded RAG query."""
    query: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Grounded AI generated answer")
    total_retrieved_chunks: int = Field(..., description="Number of context chunks used")
    llm_model: str = Field(..., description="Name of the LLM provider/model used")
    retrieved_chunks: List[VectorSearchResultItem] = Field(..., description="Retrieved context evidence chunks")
