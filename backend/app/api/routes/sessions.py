from fastapi import APIRouter, HTTPException, status
from typing import List
from backend.app.schemas.session import (
    SessionCreateRequest,
    SessionCreateResponse,
    SessionListItem,
    ChatSession,
    SessionChatRequest,
    SessionChatResponse,
)
from backend.app.database.session_store import session_store
from backend.app.services.chat.service import chat_service

router = APIRouter(prefix="/sessions", tags=["Chat Sessions & History"])


@router.post(
    "",
    response_model=SessionCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new chat session (Phase 12)"
)
async def create_session(payload: SessionCreateRequest):
    """Creates a new multi-turn conversation session with optional document scope."""
    session = session_store.create_session(title=payload.title, document_id=payload.document_id)
    return SessionCreateResponse(
        session_id=session.session_id,
        title=session.title,
        document_id=session.document_id,
        created_at=session.created_at
    )


@router.get(
    "",
    response_model=List[SessionListItem],
    status_code=status.HTTP_200_OK,
    summary="List all chat sessions (Phase 12)"
)
async def list_sessions():
    """Returns metadata and turn counts for all stored chat sessions."""
    return session_store.list_sessions()


@router.get(
    "/{session_id}",
    response_model=ChatSession,
    status_code=status.HTTP_200_OK,
    summary="Get chat session details and turns (Phase 12)"
)
async def get_session(session_id: str):
    """Retrieves full session details including all conversation turns."""
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found."
        )
    return session


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a chat session (Phase 12)"
)
async def delete_session(session_id: str):
    """Deletes a chat session and its persisted history."""
    success = session_store.delete_session(session_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found."
        )
    return {"status": "success", "message": f"Chat session {session_id} deleted successfully."}


@router.post(
    "/{session_id}/chat",
    response_model=SessionChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a chat message in a session with query rewriting and verified RAG (Phase 12)"
)
async def chat_in_session(session_id: str, payload: SessionChatRequest):
    """Sends a message/follow-up in a chat session, executing query rewriting, RAG retrieval, and verification."""
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found."
        )

    response = chat_service.chat(
        session_id=session_id,
        query=payload.query,
        top_k=payload.top_k or 5,
        use_chain_of_thought=payload.use_chain_of_thought
    )
    return response


@router.get(
    "/{session_id}/history",
    status_code=status.HTTP_200_OK,
    summary="Get session chat history (Phase 12)"
)
async def get_session_history(session_id: str):
    """Retrieves the conversation turns history for a session."""
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found."
        )
    return {"session_id": session.session_id, "turns": session.turns, "turn_count": len(session.turns)}
