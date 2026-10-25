import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.retrieval.hybrid_service import (
    hybrid_search_service,
    reciprocal_rank_fusion
)
from backend.app.services.retrieval.bm25_service import bm25_search_service
from backend.app.services.retrieval.vector_service import vector_search_service
from backend.app.database.chunk_store import chunk_store
from backend.app.database.document_store import document_store


@pytest.fixture(scope="module")
def test_client():
    """Shared test client for all hybrid tests."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_all_stores():
    """Clear all stores before each test."""
    document_store.clear()
    chunk_store.clear()
    vector_search_service.vector_store.clear()
    bm25_search_service.clear()
    yield
    document_store.clear()
    chunk_store.clear()
    vector_search_service.vector_store.clear()
    bm25_search_service.clear()


def test_reciprocal_rank_fusion_empty():
    """Test RRF with empty results."""
    results = reciprocal_rank_fusion([], [])
    assert results == []


def test_reciprocal_rank_fusion_vector_only():
    """Test RRF with only vector results."""
    from backend.app.models.search import SearchResult

    vector_results = [
        SearchResult(
            chunk_id="c1", document_id="d1", filename="f1.txt",
            page_number=1, chunk_index=0, text="test", score=0.9, source="path"
        ),
        SearchResult(
            chunk_id="c2", document_id="d1", filename="f1.txt",
            page_number=1, chunk_index=1, text="test two", score=0.8, source="path"
        ),
    ]
    results = reciprocal_rank_fusion(vector_results, [])
    assert len(results) == 2
    assert results[0].chunk_id == "c1"  # First result should be first in vector


def test_reciprocal_rank_fusion_bm25_only():
    """Test RRF with only BM25 results."""
    from backend.app.models.search import SearchResult

    bm25_results = [
        SearchResult(
            chunk_id="c1", document_id="d1", filename="f1.txt",
            page_number=1, chunk_index=0, text="test", score=0.5, source="path"
        ),
    ]
    results = reciprocal_rank_fusion([], bm25_results)
    assert len(results) == 1
    assert results[0].chunk_id == "c1"


def test_reciprocal_rank_fusion_combined():
    """Test RRF combining vector and BM25 results."""
    from backend.app.models.search import SearchResult

    # Vector search: c1=1st, c2=2nd
    vector_results = [
        SearchResult(
            chunk_id="c1", document_id="d1", filename="f1.txt",
            page_number=1, chunk_index=0, text="Python programming", score=0.9, source="path"
        ),
        SearchResult(
            chunk_id="c2", document_id="d1", filename="f1.txt",
            page_number=1, chunk_index=1, text="Java programming", score=0.8, source="path"
        ),
    ]

    # BM25 search: c2=1st (has "programming"), c1=2nd
    bm25_results = [
        SearchResult(
            chunk_id="c2", document_id="d1", filename="f1.txt",
            page_number=1, chunk_index=1, text="Java programming", score=0.7, source="path"
        ),
        SearchResult(
            chunk_id="c1", document_id="d1", filename="f1.txt",
            page_number=1, chunk_index=0, text="Python programming", score=0.6, source="path"
        ),
    ]

    results = reciprocal_rank_fusion(vector_results, bm25_results, top_k=5)

    # Both c1 and c2 appear in both lists, so both should have high RRF scores
    assert len(results) == 2
    # c2 appears at rank 1 in BM25 and rank 2 in vector = 1/61 + 1/62 = good score
    # c1 appears at rank 1 in vector and rank 2 in BM25 = similar score


def test_reciprocal_rank_fusion_different_results():
    """Test RRF when vector and BM25 return different chunks."""
    from backend.app.models.search import SearchResult

    # Vector finds semantic matches
    vector_results = [
        SearchResult(
            chunk_id="c1", document_id="d1", filename="f1.txt",
            page_number=1, chunk_index=0, text="machine learning AI", score=0.9, source="path"
        ),
        SearchResult(
            chunk_id="c2", document_id="d1", filename="f1.txt",
            page_number=1, chunk_index=1, text="deep neural networks", score=0.8, source="path"
        ),
    ]

    # BM25 finds exact keyword matches
    bm25_results = [
        SearchResult(
            chunk_id="c3", document_id="d1", filename="f1.txt",
            page_number=1, chunk_index=2, text="Python programming language", score=0.7, source="path"
        ),
    ]

    results = reciprocal_rank_fusion(vector_results, bm25_results, top_k=5)

    # Should have all 3 unique chunks
    assert len(results) == 3
    chunk_ids = [r.chunk_id for r in results]
    assert "c1" in chunk_ids
    assert "c2" in chunk_ids
    assert "c3" in chunk_ids


def test_hybrid_search_no_query(test_client):
    """Test hybrid search with empty query."""
    response = test_client.post(
        "/api/v1/hybrid/search",
        json={"query": "   ", "top_k": 5}
    )
    assert response.status_code == 200
    assert response.json()["total_results"] == 0


def test_hybrid_search_no_index(test_client):
    """Test hybrid search when nothing is indexed."""
    response = test_client.post(
        "/api/v1/hybrid/search",
        json={"query": "test", "top_k": 5}
    )
    assert response.status_code == 200
    assert response.json()["total_results"] == 0


def test_hybrid_search_full_workflow(test_client):
    """Test complete hybrid search workflow."""
    # 1. Upload document
    test_content = """
    Python is a high-level programming language.
    Python is widely used in machine learning.
    BM25 is a ranking function for information retrieval.
    Vector search uses embeddings for semantic similarity.
    """

    response = test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.txt", test_content, "text/plain")}
    )
    assert response.status_code == 201
    document_id = response.json()["document_id"]

    # 2. Process chunks
    response = test_client.post(f"/api/v1/chunks/process/{document_id}?chunk_size=100&chunk_overlap=20")
    assert response.status_code == 200

    # 3. Index in vector store
    response = test_client.post(f"/api/v1/vector-store/index/{document_id}")
    assert response.status_code == 200

    # 4. Build BM25 index
    response = test_client.post(f"/api/v1/bm25/index/{document_id}")
    assert response.status_code == 200

    # 5. Hybrid search for "Python"
    response = test_client.post(
        "/api/v1/hybrid/search",
        json={"query": "Python", "top_k": 5}
    )
    assert response.status_code == 200
    search_data = response.json()
    assert search_data["total_results"] > 0
    assert search_data["query"] == "Python"

    # Check that at least some results contain Python-related content
    # (BM25 should find Python chunks, vector search may return different results with mock embeddings)
    results = search_data["results"]
    python_results = [r for r in results if "python" in r["text"].lower()]
    assert len(python_results) > 0, "At least one result should contain 'Python'"


def test_hybrid_search_scoped_to_document(test_client):
    """Test hybrid search with document_id scope filter."""
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
    test_client.post(f"/api/v1/vector-store/index/{doc1_id}")
    test_client.post(f"/api/v1/bm25/index/{doc1_id}")

    test_client.post(f"/api/v1/chunks/process/{doc2_id}")
    test_client.post(f"/api/v1/vector-store/index/{doc2_id}")
    test_client.post(f"/api/v1/bm25/index/{doc2_id}")

    # Search only in doc1 for "Python"
    response = test_client.post(
        "/api/v1/hybrid/search",
        json={"query": "Python", "top_k": 5, "document_id": doc1_id}
    )
    assert response.status_code == 200
    search_data = response.json()
    assert search_data["total_results"] > 0
    for result in search_data["results"]:
        assert result["document_id"] == doc1_id


def test_hybrid_stats(test_client):
    """Test hybrid stats endpoint."""
    response = test_client.get("/api/v1/hybrid/stats")
    assert response.status_code == 200
    stats = response.json()
    assert "hybrid" in stats
    assert "vector_search" in stats
    assert "bm25_search" in stats
    assert stats["fusion_method"] == "RRF (Reciprocal Rank Fusion)"