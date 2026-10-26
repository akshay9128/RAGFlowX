import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.citations.service import verification_service
from backend.app.models.search import VectorSearchResult
from backend.app.schemas.search import VectorSearchResultItem


def test_claim_extraction():
    text = "RAGFlowX is an advanced RAG engine. It supports hybrid search. Is it fast?"
    claims = verification_service.extract_claims(text)
    assert len(claims) == 3
    assert claims[0] == "RAGFlowX is an advanced RAG engine."
    assert claims[1] == "It supports hybrid search."
    assert claims[2] == "Is it fast?"


def test_verification_service_supported_and_unsupported():
    chunk = VectorSearchResult(
        chunk_id="c1",
        document_id="doc1",
        filename="manual.pdf",
        page_number=1,
        chunk_index=0,
        text="RAGFlowX uses Reciprocal Rank Fusion and Cross-Encoder reranking for superior search results.",
        score=0.95,
        source="vector"
    )

    # Supported claim
    claim_supported = "RAGFlowX uses Reciprocal Rank Fusion for search."
    res_supported = verification_service.verify_claim(claim_supported, [chunk])
    assert res_supported.support_status in ("supported", "partially_supported")
    assert res_supported.support_score > 0.0

    # Unsupported claim
    claim_unsupported = "Quantum computing powers the central processing core of the system."
    res_unsupported = verification_service.verify_claim(claim_unsupported, [chunk])
    assert res_unsupported.support_status == "unsupported"
    assert res_unsupported.support_score < 0.15


def test_standalone_verification_endpoint(client: TestClient):
    chunk_item = VectorSearchResultItem(
        chunk_id="c1",
        document_id="doc1",
        filename="manual.pdf",
        page_number=1,
        chunk_index=0,
        text="The retrieval system integrates BM25 keyword search and dense vector search.",
        score=0.9,
        source="vector"
    )

    payload = {
        "answer": "The retrieval system integrates BM25 keyword search and dense vector search. Magic lasers control the server.",
        "chunks": [chunk_item.model_dump()]
    }

    response = client.post("/api/v1/rag/verify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_claims"] == 2
    assert len(data["unsupported_claims"]) == 1
    assert data["groundedness_score"] > 0.0
    assert "Magic lasers" in data["unsupported_claims"][0]


def test_query_verified_rag_endpoint(client: TestClient):
    payload = {
        "query": "What is RAGFlowX?",
        "top_k": 3,
        "use_chain_of_thought": False
    }

    response = client.post("/api/v1/rag/query/verified", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "original_answer" in data
    assert "verified_answer" in data
    assert "claims" in data
    assert "citations" in data
    assert isinstance(data["groundedness_score"], float)
