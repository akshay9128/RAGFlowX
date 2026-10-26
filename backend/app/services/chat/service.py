from typing import List, Optional, Tuple
from datetime import datetime
import uuid
from backend.app.schemas.session import ChatSession, ChatTurn, SessionChatResponse
from backend.app.database.session_store import session_store
from backend.app.services.retrieval.rag_service import rag_service
from backend.app.schemas.rag import Citation
from backend.app.schemas.verification import VerifiedClaim


class ChatService:
    """Service managing chat sessions, query rewriting, memory pruning, and RAG chat execution."""

    def __init__(self):
        self.max_turns_window = 10  # Sliding window memory limit

    def rewrite_query(self, query: str, turns: List[ChatTurn]) -> str:
        """Rewrites a follow-up query into a standalone search query using recent conversation history."""
        if not turns:
            return query

        query_lower = query.lower().strip()
        follow_up_markers = ["it", "that", "this", "they", "he", "she", "why", "how", "what about", "more details", "explain"]
        is_follow_up = any(marker in query_lower.split() for marker in follow_up_markers) or len(query.split()) < 4

        if not is_follow_up:
            return query

        recent_turns = turns[-4:]
        last_user_query = next((t.content for t in reversed(recent_turns) if t.role == "user"), "")
        if last_user_query and last_user_query.lower() != query_lower:
            rewritten = f"{last_user_query} {query}"
            return rewritten

        return query

    def prune_history(self, session: ChatSession) -> ChatSession:
        """Applies sliding window pruning to keep conversation turns within memory limits."""
        if len(session.turns) > self.max_turns_window * 2:
            session.turns = session.turns[-(self.max_turns_window * 2):]
        return session

    def chat(
        self,
        session_id: str,
        query: str,
        top_k: int = 5,
        use_chain_of_thought: bool = False
    ) -> SessionChatResponse:
        """Executes a multi-turn chat interaction with query rewriting, RAG, and history updates."""
        session = session_store.get_session(session_id)
        if not session:
            session = session_store.create_session(title="Auto Session")

        rewritten_query = self.rewrite_query(query, session.turns)

        rag_response = rag_service.query_verified_rag(
            query=rewritten_query,
            top_k=top_k,
            document_id=session.document_id,
            use_chain_of_thought=use_chain_of_thought
        )

        now = datetime.utcnow().isoformat()
        user_turn_id = str(uuid.uuid4())
        assistant_turn_id = str(uuid.uuid4())

        user_turn = ChatTurn(
            turn_id=user_turn_id,
            role="user",
            content=query,
            timestamp=now
        )
        session.turns.append(user_turn)

        assistant_turn = ChatTurn(
            turn_id=assistant_turn_id,
            role="assistant",
            content=rag_response.original_answer,
            timestamp=datetime.utcnow().isoformat(),
            citations=rag_response.citations,
            groundedness_score=rag_response.groundedness_score,
            claims=rag_response.claims
        )
        session.turns.append(assistant_turn)

        session = self.prune_history(session)
        session_store.save_session(session)

        return SessionChatResponse(
            session_id=session.session_id,
            query=query,
            rewritten_query=rewritten_query if rewritten_query != query else None,
            answer=rag_response.original_answer,
            verified_answer=rag_response.verified_answer,
            citations=rag_response.citations,
            groundedness_score=rag_response.groundedness_score,
            claims=rag_response.claims,
            turn_count=len(session.turns)
        )


chat_service = ChatService()
