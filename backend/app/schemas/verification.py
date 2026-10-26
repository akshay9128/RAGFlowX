from typing import List, Optional
from pydantic import BaseModel, Field
from backend.app.schemas.rag import Citation
from backend.app.schemas.search import VectorSearchResultItem


class VerifiedClaim(BaseModel):
    """An atomic claim extracted from the answer and verified against retrieved chunks."""
    claim_text: str = Field(..., description="Atomic sentence or fact extracted from the generated answer")
    support_status: str = Field(..., description="Support status: 'supported', 'partially_supported', or 'unsupported'")
    support_score: float = Field(..., description="Semantic/lexical support score between 0.0 and 1.0")
    supporting_chunk_id: Optional[str] = Field(default=None, description="ID of the best matching source chunk")
    supporting_citation: Optional[Citation] = Field(default=None, description="Explicit citation for the supporting chunk")
    reason: Optional[str] = Field(default=None, description="Explanation of verification result")


class VerifiedAnswerRequest(BaseModel):
    """API request for verified RAG query."""
    query: str = Field(..., description="User question or prompt", min_length=1)
    top_k: Optional[int] = Field(default=5, description="Number of top relevant context chunks to retrieve", gt=0, le=50)
    document_id: Optional[str] = Field(default=None, description="Optional document ID to restrict query scope")
    use_chain_of_thought: bool = Field(default=False, description="Enable chain-of-thought reasoning")


class VerifiedAnswerResponse(BaseModel):
    """API response for verified RAG query with claim-level attribution."""
    query: str = Field(..., description="Original user question")
    original_answer: str = Field(..., description="Raw LLM generated answer")
    verified_answer: str = Field(..., description="Refined answer where every sentence is cited or marked unverified")
    total_claims: int = Field(..., description="Total atomic claims extracted")
    supported_claims_count: int = Field(..., description="Number of fully supported claims")
    groundedness_score: float = Field(..., description="Overall answer groundedness score (0.0 to 1.0)")
    claims: List[VerifiedClaim] = Field(..., description="Detailed verification results per atomic claim")
    unsupported_claims: List[str] = Field(..., description="List of unsupported or flagged claim texts")
    citations: List[Citation] = Field(..., description="All source citations used")
    llm_model: str = Field(..., description="LLM model used")


class StandaloneVerificationRequest(BaseModel):
    """API request to verify any answer text against given evidence chunks."""
    answer: str = Field(..., description="Answer text to verify", min_length=1)
    chunks: List[VectorSearchResultItem] = Field(..., description="Evidence chunks to verify against")


class StandaloneVerificationResponse(BaseModel):
    """API response for standalone answer verification."""
    answer: str = Field(..., description="Answer text verified")
    verified_answer: str = Field(..., description="Annotated answer with verification markers")
    groundedness_score: float = Field(..., description="Overall groundedness score")
    total_claims: int = Field(..., description="Total claims analyzed")
    claims: List[VerifiedClaim] = Field(..., description="Claim verification details")
    unsupported_claims: List[str] = Field(..., description="Unsupported claim texts")
