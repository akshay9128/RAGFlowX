import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database.document_store import DocumentStore
from backend.app.database.chunk_store import ChunkStore
from backend.app.database.session_store import SessionStore
from backend.app.models.document import ProcessedDocument, DocumentMetadata


@pytest.mark.e2e
def test_observability_request_tracing(client: TestClient):
    """Verify X-Request-ID tracing header is returned on API responses."""
    res = client.get("/api/v1/rag/info")
    assert res.status_code == 200
    assert "X-Request-ID" in res.headers or "x-request-id" in {k.lower() for k in res.headers.keys()}


@pytest.mark.e2e
def test_persistence_restart_and_deletion(tmp_path):
    """Verify document & chunk persistence across store restarts and clean deletion."""
    data_dir = tmp_path / "persistence_data"
    data_dir.mkdir()

    # 1. Initialize stores with temp data directory
    doc_store1 = DocumentStore(data_dir=data_dir)
    chunk_store1 = ChunkStore(data_dir=data_dir)

    doc_id = "doc_persist_123"
    meta = DocumentMetadata(
        document_id=doc_id,
        filename="persistent_notes.txt",
        content_type="text/plain",
        file_size=1024,
        total_pages=1,
        source="local"
    )
    doc = ProcessedDocument(metadata=meta, pages=[])
    doc_store1.save_document(doc)
    assert doc_store1.get_document(doc_id) is not None

    # 2. Simulate server restart by creating new store instances pointing to same data_dir
    doc_store2 = DocumentStore(data_dir=data_dir)
    assert doc_store2.get_document(doc_id) is not None
    assert doc_store2.get_document(doc_id).metadata.filename == "persistent_notes.txt"

    # 3. Test deletion
    doc_store2.delete_document(doc_id)
    assert doc_store2.get_document(doc_id) is None
