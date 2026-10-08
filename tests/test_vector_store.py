import pytest
from backend.app.services.embeddings.mock import MockEmbeddingProvider
from backend.app.services.embeddings.service import embedding_service
from backend.app.database.document_store import document_store
from backend.app.database.chunk_store import chunk_store
from backend.app.services.retrieval.vector_service import vector_search_service


@pytest.fixture(autouse=True)
def setup_mock_environment():
    """Ensure clean stores and mock embedding provider for fast vector tests."""
    document_store.clear()
    chunk_store.clear()
    vector_search_service.vector_store.clear()
    original_provider = embedding_service.provider
    mock_prov = MockEmbeddingProvider(dim=384)
    embedding_service.set_provider(mock_prov)
    yield
    embedding_service.set_provider(original_provider)
    document_store.clear()
    chunk_store.clear()
    vector_search_service.vector_store.clear()


def test_vector_store_stats_api(client):
    res = client.get("/api/v1/vector-store/stats")
    assert res.status_code == 200
    data = res.json()
    assert "provider" in data
    assert "total_vectors" in data
    assert "dimension" in data


def test_vector_store_end_to_end_workflow(client):
    # 1. Upload document (Phase 2)
    doc_content = b"Vector databases store high dimensional embeddings for fast nearest neighbor search."
    upload_res = client.post(
        "/api/v1/documents/upload",
        files={"file": ("vector_guide.txt", doc_content, "text/plain")}
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]

    # 2. Process text chunks (Phase 3)
    chunk_res = client.post(f"/api/v1/chunks/process/{doc_id}?chunk_size=100&chunk_overlap=10")
    assert chunk_res.status_code == 200

    # 3. Index chunks into Vector Database (Phase 5)
    index_res = client.post(f"/api/v1/vector-store/index/{doc_id}")
    assert index_res.status_code == 200
    assert index_res.json()["indexed_chunks"] >= 1

    # 4. Check vector stats
    stats_res = client.get("/api/v1/vector-store/stats")
    assert stats_res.json()["total_vectors"] >= 1

    # 5. Perform Similarity Search
    search_res = client.post(
        "/api/v1/vector-store/search",
        json={
            "query": "What is nearest neighbor search in vector databases?",
            "top_k": 3
        }
    )
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert search_data["total_results"] >= 1
    top_result = search_data["results"][0]
    assert top_result["document_id"] == doc_id
    assert top_result["filename"] == "vector_guide.txt"
    assert "score" in top_result
    assert 0.0 <= top_result["score"] <= 1.0

    # 6. Delete document vectors
    del_res = client.delete(f"/api/v1/vector-store/document/{doc_id}")
    assert del_res.status_code == 200

    # 7. Confirm deletion
    stats_after = client.get("/api/v1/vector-store/stats")
    assert stats_after.json()["total_vectors"] == 0


def test_search_empty_store(client):
    res = client.post(
        "/api/v1/vector-store/search",
        json={"query": "test query", "top_k": 5}
    )
    assert res.status_code == 200
    assert res.json()["total_results"] == 0
