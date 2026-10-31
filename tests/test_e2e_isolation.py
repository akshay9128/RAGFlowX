import os
import secrets
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.config import settings
from backend.app.database.document_store import document_store, DocumentStore
from backend.app.database.chunk_store import chunk_store, ChunkStore
from backend.app.database.session_store import session_store, SessionStore
from backend.app.core.security import api_key_store, APIKeyStore


@pytest.mark.e2e
def test_e2e_temp_directory_isolation(tmp_path):
    """Prove that e2e stores can be initialized with isolated temp data and upload directories."""
    temp_data = tmp_path / "data"
    temp_uploads = tmp_path / "uploads"
    temp_data.mkdir()
    temp_uploads.mkdir()

    doc_store = DocumentStore(data_dir=temp_data)
    chk_store = ChunkStore(data_dir=temp_data)
    sess_store = SessionStore(data_dir=temp_data)

    assert doc_store.data_dir == temp_data
    assert chk_store.data_dir == temp_data
    assert sess_store.data_dir == temp_data
    assert len(doc_store.list_documents()) == 0


@pytest.mark.e2e
def test_auth_on_per_request_enforcement(monkeypatch, tmp_path):
    """Prove that settings.AUTH_ENABLED is read per-request and enforces 401 when enabled without a key."""
    temp_data = tmp_path / "data"
    temp_data.mkdir()

    # Configure Auth ON and runtime pepper
    pepper = secrets.token_hex(32)
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    monkeypatch.setattr(settings, "API_KEY_PEPPER", pepper)

    # Re-initialize api_key_store with temp data dir
    local_key_store = APIKeyStore(data_dir=temp_data)
    monkeypatch.setattr("backend.app.core.security.api_key_store", local_key_store)

    client = TestClient(app)

    # 1. Request to protected endpoint without API key -> should get 401
    response = client.get(f"{settings.API_PREFIX}/rag/info")
    assert response.status_code == 401
    assert "Missing API key" in response.text

    # 2. Create valid API key
    raw_key, record = local_key_store.create_key(name="Test Admin Key", scopes=["read", "write", "admin"])

    # 3. Request with valid API key in X-API-Key header -> should get 200
    response_with_key = client.get(
        f"{settings.API_PREFIX}/rag/info",
        headers={"X-API-Key": raw_key}
    )
    assert response_with_key.status_code == 200
    data = response_with_key.json()
    assert "llm_provider" in data
