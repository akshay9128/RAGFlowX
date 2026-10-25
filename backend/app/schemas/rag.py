from typing import List, Optional
from pydantic import BaseModel, Field
from backend.app.schemas.search import VectorSearchResultItem


class Citation(BaseModel):
    """A verified source citation tagging a claim in the generated answer."""
    document_id: str = Field(..., description="ID of the source document")
    filename: str = Field(..., description="Name of the source document")
    page_number: int = Field(..., description="1-indexed page number in the source document")
    chunk_index: int = Field(..., description="Sequence index of the chunk in the document")
    quoted_text: str = Field(..., description="Exact chunk text cited in the answer")


class RAGQueryRequest(BaseModel):
    """API request schema for grounded RAG query."""
    query: str = Field(..., description="User question or prompt", min_length=1)
    top_k: Optional[int] = Field(default=5, description="Number of top relevant context chunks to retrieve", gt=0, le=50)
    document_id: Optional[str] = Field(default=None, description="Optional document ID to restrict query scope")
    use_chain_of_thought: bool = Field(default=False, description="Enable chain-of-thought reasoning in the prompt")
    use_citations: bool = Field(default=False, description="Attach explicit source citations to the answer")
    use_confidence_scoring: bool = Field(default=False, description="Include a confidence score in the response")


class RAGQueryResponse(BaseModel):
    """API response schema for grounded RAG query."""
    query: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Grounded AI generated answer")
    total_retrieved_chunks: int = Field(..., description="Number of context chunks used")
    llm_model: str = Field(..., description="Name of the LLM provider/model used")
    retrieved_chunks: List[VectorSearchResultItem] = Field(..., description="Retrieved context evidence chunks")
    citations: Optional[List[Citation]] = Field(default=None, description="Source citations for claims in the answer")
    confidence_score: Optional[float] = Field(default=None, description="Confidence score between 0.0 and 1.0")
    reasoning_trace: Optional[str] = Field(default=None, description="Chain-of-thought reasoning trace")
