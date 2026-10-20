import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.retrieval.bm25_service import bm25_search_service
from backend.app.database.chunk_store import chunk_store


@pytest.fixture(scope="module")
def test_client():
    """Shared test client for all BM25 tests."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_bm25_index():
    """Clear BM25 index before each test."""
    bm25_search_service.clear()
    chunk_store.clear()
    yield
    bm25_search_service.clear()
    chunk_store.clear()


def test_bm25_index_document(test_client):
    """Test building BM25 index for a document."""
    # First, upload and chunk a document
    # Create a simple test document
    test_content = """
    Python is a high-level programming language.
    Python supports multiple programming paradigms.
    Python is interpreted and dynamic.
    Machine learning uses Python extensively.
    Deep learning frameworks like PyTorch are popular.
    """

    # Upload document
    response = test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("test_doc.txt", test_content, "text/plain")}
    )
    assert response.status_code == 201  # 201 Created is correct for POST
    upload_data = response.json()
    document_id = upload_data["document_id"]

    # Process into chunks
    response = test_client.post(f"/api/v1/chunks/process/{document_id}")
    assert response.status_code == 200
    chunk_data = response.json()
    assert chunk_data["total_chunks"] > 0

    # Build BM25 index
    response = test_client.post(f"/api/v1/bm25/index/{document_id}")
    assert response.status_code == 200
    index_data = response.json()
    assert index_data["indexed_chunks"] > 0
    assert index_data["document_id"] == document_id


def test_bm25_search(test_client):
    """Test BM25 keyword search functionality."""
    # Setup: upload and index document
    test_content = """
    Apple is a fruit.
    Apple Inc. makes iPhones.
    Apple computers are popular.
    The apple tree produces fruit.
    """
    response = test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("apple.txt", test_content, "text/plain")}
    )
    document_id = response.json()["document_id"]

    # Process and index
    test_client.post(f"/api/v1/chunks/process/{document_id}")
    test_client.post(f"/api/v1/bm25/index/{document_id}")

    # Search for "Apple" - should find all apple-related chunks
    response = test_client.post(
        "/api/v1/bm25/search",
        json={"query": "Apple", "top_k": 5}
    )
    assert response.status_code == 200
    search_data = response.json()
    assert search_data["total_results"] > 0
    assert search_data["query"] == "Apple"

    # Check that results contain relevant content
    results = search_data["results"]
    for result in results:
        assert "apple" in result["text"].lower() or "Apple" in result["text"]


def test_bm25_search_scoped_to_document(test_client):
    """Test BM25 search with document_id scope filter."""
    # Upload two documents
    doc1_content = "Python programming language is great."
    doc2_content = "Java is another programming language."

    response1 = test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("doc1.txt", doc1_content, "text/plain")}
    )
    doc1_id = response1.json()["document_id"]

    response2 = test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("doc2.txt", doc2_content, "text/plain")}
    )
    doc2_id = response2.json()["document_id"]

    # Process and index both
    test_client.post(f"/api/v1/chunks/process/{doc1_id}")
    test_client.post(f"/api/v1/bm25/index/{doc1_id}")
    test_client.post(f"/api/v1/chunks/process/{doc2_id}")
    test_client.post(f"/api/v1/bm25/index/{doc2_id}")

    # Search only in doc1 for "Python"
    response = test_client.post(
        "/api/v1/bm25/search",
        json={"query": "Python", "top_k": 5, "document_id": doc1_id}
    )
    assert response.status_code == 200
    search_data = response.json()
    assert search_data["total_results"] > 0
    for result in search_data["results"]:
        assert result["document_id"] == doc1_id


def test_bm25_search_no_query(test_client):
    """Test BM25 search with empty query returns no results."""
    response = test_client.post(
        "/api/v1/bm25/search",
        json={"query": "   ", "top_k": 5}
    )
    # Should still return 200 with empty results
    assert response.status_code == 200
    assert response.json()["total_results"] == 0


def test_bm25_search_no_index(test_client):
    """Test BM25 search when no document has been indexed."""
    response = test_client.post(
        "/api/v1/bm25/search",
        json={"query": "test", "top_k": 5}
    )
    assert response.status_code == 200
    assert response.json()["total_results"] == 0


def test_bm25_index_nonexistent_document(test_client):
    """Test BM25 index for document that doesn't exist."""
    response = test_client.post("/api/v1/bm25/index/nonexistent-doc-id")
    assert response.status_code == 404


def test_bm25_delete_document(test_client):
    """Test deleting BM25 index for a document."""
    # Setup
    test_content = "Test document content for deletion."
    response = test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.txt", test_content, "text/plain")}
    )
    document_id = response.json()["document_id"]

    # Process and index
    test_client.post(f"/api/v1/chunks/process/{document_id}")
    test_client.post(f"/api/v1/bm25/index/{document_id}")

    # Verify index exists
    stats_response = test_client.get("/api/v1/bm25/stats")
    assert stats_response.json()["indexed_documents"] > 0

    # Delete
    response = test_client.delete(f"/api/v1/bm25/document/{document_id}")
    assert response.status_code == 200

    # Verify deletion
    stats_response = test_client.get("/api/v1/bm25/stats")
    assert stats_response.json()["indexed_documents"] == 0


def test_bm25_delete_nonexistent_document(test_client):
    """Test deleting BM25 index for nonexistent document."""
    response = test_client.delete("/api/v1/bm25/document/nonexistent-doc-id")
    assert response.status_code == 404


def test_bm25_stats(test_client):
    """Test BM25 statistics endpoint."""
    # Should return empty stats when nothing indexed
    response = test_client.get("/api/v1/bm25/stats")
    assert response.status_code == 200
    stats = response.json()
    assert stats["provider"] == "bm25"
    assert "total_indexed_chunks" in stats
    assert "indexed_documents" in stats