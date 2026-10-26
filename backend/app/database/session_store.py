import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import uuid
from backend.app.config import settings
from backend.app.schemas.session import ChatSession, ChatTurn, SessionListItem


class SessionStore:
    """Persistent storage manager for chat sessions and conversation history."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: Dict[str, ChatSession] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        """Loads saved chat sessions from DATA_DIR."""
        for json_file in self.data_dir.glob("session_*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    session = ChatSession.model_validate(data)
                    self._sessions[session.session_id] = session
            except Exception as exc:
                print(f"Warning: Failed to load chat session from {json_file}: {exc}")

    def create_session(self, title: Optional[str] = "New Chat Session", document_id: Optional[str] = None) -> ChatSession:
        """Creates a new chat session."""
        session_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        session = ChatSession(
            session_id=session_id,
            title=title or "New Chat Session",
            document_id=document_id,
            turns=[],
            created_at=now,
            updated_at=now
        )
        return self.save_session(session)

    def save_session(self, session: ChatSession) -> ChatSession:
        """Saves a chat session in-memory and to disk."""
        session.updated_at = datetime.utcnow().isoformat()
        self._sessions[session.session_id] = session
        file_path = self.data_dir / f"session_{session.session_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(session.model_dump_json(indent=2))
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Retrieves a chat session by session_id."""
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[SessionListItem]:
        """Lists metadata and turn counts for all stored chat sessions."""
        items = []
        for s in self._sessions.values():
            items.append(SessionListItem(
                session_id=s.session_id,
                title=s.title,
                document_id=s.document_id,
                turn_count=len(s.turns),
                created_at=s.created_at,
                updated_at=s.updated_at
            ))
        items.sort(key=lambda x: x.updated_at, reverse=True)
        return items

    def delete_session(self, session_id: str) -> bool:
        """Deletes a chat session by session_id."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            file_path = self.data_dir / f"session_{session_id}.json"
            if file_path.exists():
                file_path.unlink()
            return True
        return False

    def clear(self):
        """Clears all stored sessions (for testing)."""
        self._sessions.clear()
        for json_file in self.data_dir.glob("session_*.json"):
            try:
                json_file.unlink()
            except Exception:
                pass


# Global singleton instance
session_store = SessionStore()
