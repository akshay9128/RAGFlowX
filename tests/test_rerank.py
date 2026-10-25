import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.reranking.mock import MockReranker
from backend.app.services.reranking.service import rerank_service
from backend.app.models.search import SearchResult


@pytest.fixture(scope="module")
def test_client():
    return TestClient(app)


def test_mock_reranker():
    reranker = MockReranker()
    docs = [
        "This is about Python programming.",
        "Completely unrelated text about cooking.",
        "Advanced Python machine learning guide."
    ]
    results = reranker.rerank("Python", docs, top_k=2)
    assert len(results) == 2
    top_indices = [r[0] for r in results]
    assert 0 in top_indices or 2 in top_indices


def test_rerank_api(test_client):
    response = test_client.post(
        "/api/v1/rerank/",
        json={
            "query": "fastapi",
            "documents": [
                "FastAPI is a modern web framework for Python.",
                "Baking recipes for chocolate cake.",
                "FastAPI dependency injection and routing."
            ],
            "top_k": 2
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "fastapi"
    assert data["total_results"] == 2
    assert len(data["results"]) == 2
