from typing import List, Optional, Any
from pydantic import BaseModel, Field
from backend.app.schemas.rag import Citation
from backend.app.schemas.verification import VerifiedClaim


class ChatTurn(BaseModel):
    """A single turn (user query and assistant response) in a chat session."""
    turn_id: str = Field(..., description="Unique turn identifier")
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message text")
    timestamp: str = Field(..., description="ISO timestamp")
    citations: Optional[List[Citation]] = Field(default=None, description="Source citations if assistant response")
    groundedness_score: Optional[float] = Field(default=None, description="Groundedness score if assistant response")
    claims: Optional[List[VerifiedClaim]] = Field(default=None, description="Verified claims if assistant response")


class ChatSession(BaseModel):
    """A multi-turn chat session."""
    session_id: str = Field(..., description="Unique session ID")
    title: str = Field(..., description="Session title")
    document_id: Optional[str] = Field(default=None, description="Optional document scope restriction")
    turns: List[ChatTurn] = Field(default_factory=list, description="Conversation turns history")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")


class SessionCreateRequest(BaseModel):
    """Request to create a new chat session."""
    title: Optional[str] = Field(default="New Chat Session", description="Session title")
    document_id: Optional[str] = Field(default=None, description="Optional document ID scope")


class SessionCreateResponse(BaseModel):
    """Response after creating a chat session."""
    session_id: str
    title: str
    document_id: Optional[str]
    created_at: str


class SessionListItem(BaseModel):
    """Summary item for listing sessions."""
    session_id: str
    title: str
    document_id: Optional[str]
    turn_count: int
    created_at: str
    updated_at: str


class SessionChatRequest(BaseModel):
    """Request to send a chat message in a session."""
    query: str = Field(..., description="User query or follow-up", min_length=1)
    top_k: Optional[int] = Field(default=5, description="Number of chunks to retrieve", gt=0, le=50)
    use_chain_of_thought: bool = Field(default=False, description="Enable chain-of-thought reasoning")
    stream: bool = Field(default=False, description="Enable streaming response")


class SessionChatResponse(BaseModel):
    """Response from chat message execution."""
    session_id: str
    query: str
    rewritten_query: Optional[str] = Field(default=None, description="Query rewritten using history if follow-up")
    answer: str = Field(..., description="Assistant answer")
    verified_answer: Optional[str] = Field(default=None, description="Verified annotated answer")
    citations: Optional[List[Citation]] = Field(default=None, description="Source citations")
    groundedness_score: Optional[float] = Field(default=None, description="Groundedness score")
    claims: Optional[List[VerifiedClaim]] = Field(default=None, description="Verified claims")
    turn_count: int = Field(..., description="Total turns in session after message")
