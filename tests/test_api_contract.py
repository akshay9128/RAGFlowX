import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
import io

client = TestClient(app)

def test_validation_error_format():
    response = client.post("/api/v1/rag/query", json={})
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "VALIDATION_ERROR"
    assert "request_id" in data["error"]
    assert "X-Request-ID" in response.headers

def test_not_found_error_format():
    response = client.get("/api/v1/documents/non-existent-id")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "NOT_FOUND"
    assert "request_id" in data["error"]

def test_pipeline_ingest_endpoint():
    file_content = b"This is a test document content for pipeline ingestion. It contains multiple sentences and technical terms like RAG, BM25, and embeddings."
    response = client.post(
        "/api/v1/pipeline/ingest",
        files={"file": ("pipeline_test.txt", file_content, "text/plain")},
        params={"chunk_size": 100, "chunk_overlap": 20}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "success"
    assert "document_id" in data
    assert data["filename"] == "pipeline_test.txt"
    assert data["total_chunks"] > 0
    assert data["vector_indexed"] > 0
    assert data["bm25_indexed"] > 0
