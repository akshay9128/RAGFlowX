import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database.session_store import session_store
from backend.app.services.chat.service import chat_service


@pytest.fixture(autouse=True)
def clean_sessions():
    session_store.clear()
    yield
    session_store.clear()


def test_session_lifecycle(client: TestClient):
    # 1. Create session
    create_payload = {"title": "Test Chat Session", "document_id": "doc123"}
    response = client.post("/api/v1/sessions", json=create_payload)
    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert data["title"] == "Test Chat Session"
    assert data["document_id"] == "doc123"
    session_id = data["session_id"]

    # 2. List sessions
    list_response = client.get("/api/v1/sessions")
    assert list_response.status_code == 200
    sessions = list_response.json()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == session_id

    # 3. Get session details
    get_response = client.get(f"/api/v1/sessions/{session_id}")
    assert get_response.status_code == 200
    s_data = get_response.json()
    assert s_data["session_id"] == session_id
    assert len(s_data["turns"]) == 0

    # 4. Delete session
    del_response = client.delete(f"/api/v1/sessions/{session_id}")
    assert del_response.status_code == 200

    # 5. Verify deleted
    get_after_del = client.get(f"/api/v1/sessions/{session_id}")
    assert get_after_del.status_code == 404


def test_chat_session_interaction(client: TestClient):
    # Create session
    create_payload = {"title": "RAG Chat"}
    res = client.post("/api/v1/sessions", json=create_payload)
    session_id = res.json()["session_id"]

    # Send chat message
    chat_payload = {"query": "What is hybrid search?", "top_k": 3}
    chat_res = client.post(f"/api/v1/sessions/{session_id}/chat", json=chat_payload)
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert chat_data["session_id"] == session_id
    assert chat_data["query"] == "What is hybrid search?"
    assert "answer" in chat_data
    assert chat_data["turn_count"] == 2

    # Get history
    history_res = client.get(f"/api/v1/sessions/{session_id}/history")
    assert history_res.status_code == 200
    history_data = history_res.json()
    assert history_data["turn_count"] == 2
    assert history_data["turns"][0]["role"] == "user"
    assert history_data["turns"][1]["role"] == "assistant"


def test_query_rewriting():
    session = session_store.create_session(title="Rewrite Test")
    chat_service.chat(session.session_id, query="Explain BM25 search algorithm.")

    follow_up = "How does it work?"
    session_obj = session_store.get_session(session.session_id)
    rewritten = chat_service.rewrite_query(follow_up, session_obj.turns)
    assert rewritten != follow_up
