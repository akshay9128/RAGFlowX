from typing import List, Optional
from pydantic import BaseModel, Field


class RerankRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    documents: List[str] = Field(..., description="List of document text passages to rerank")
    top_k: Optional[int] = Field(5, description="Number of top reranked items to return")


class RerankResultItem(BaseModel):
    index: int = Field(..., description="Original index of the document")
    text: str = Field(..., description="Document text passage")
    score: float = Field(..., description="Cross-encoder relevance score")


class RerankResponse(BaseModel):
    query: str
    total_results: int
    results: List[RerankResultItem]
    model_name: str
