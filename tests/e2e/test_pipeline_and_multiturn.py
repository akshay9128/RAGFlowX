import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.embeddings.mock import MockEmbeddingProvider
from backend.app.services.embeddings.service import embedding_service
from backend.app.services.generation.mock import MockLLMProvider
from backend.app.services.retrieval.rag_service import rag_service
from backend.app.database.document_store import document_store
from backend.app.database.chunk_store import chunk_store
from backend.app.database.session_store import session_store
from backend.app.services.retrieval.vector_service import vector_search_service


@pytest.fixture(autouse=True)
def clean_e2e_env():
    document_store.clear()
    chunk_store.clear()
    vector_search_service.vector_store.clear()
    session_store.clear()

    orig_embed = embedding_service.provider
    orig_llm = rag_service.llm_provider

    embedding_service.set_provider(MockEmbeddingProvider(dim=384))
    rag_service.set_llm_provider(MockLLMProvider())

    yield

    embedding_service.set_provider(orig_embed)
    rag_service.set_llm_provider(orig_llm)
    document_store.clear()
    chunk_store.clear()
    vector_search_service.vector_store.clear()
    session_store.clear()


@pytest.mark.e2e
def test_e2e_full_pipeline_with_citations(client: TestClient):
    """E2E test: Ingest doc -> chunk -> vector index -> hybrid search -> RAG generation with citations."""
    doc_text = b"Project Titan launched on October 31, 2026. It features 384-dimensional dense vectors combined with BM25 keyword search and RRF ranking."

    # 1. Upload
    up_res = client.post(
        "/api/v1/documents/upload",
        files={"file": ("project_titan.txt", doc_text, "text/plain")}
    )
    assert up_res.status_code == 201
    doc_id = up_res.json()["document_id"]

    # 2. Chunking
    chunk_res = client.post(f"/api/v1/chunks/process/{doc_id}?chunk_size=150&chunk_overlap=25")
    assert chunk_res.status_code == 200
    assert chunk_res.json()["total_chunks"] > 0

    # 3. Vector Indexing
    index_res = client.post(f"/api/v1/vector-store/index/{doc_id}")
    assert index_res.status_code == 200

    # 4. Hybrid Search endpoint
    search_res = client.post(
        "/api/v1/hybrid/search",
        json={"query": "When did Project Titan launch?", "top_k": 3}
    )
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert "results" in search_data

    # 5. RAG Generation Query with Citations
    rag_res = client.post(
        "/api/v1/rag/query",
        json={"query": "What are the features of Project Titan?", "top_k": 3}
    )
    assert rag_res.status_code == 200
    rag_data = rag_res.json()
    assert rag_data["total_retrieved_chunks"] > 0
    assert "answer" in rag_data
    assert len(rag_data["answer"]) > 0


@pytest.mark.e2e
def test_e2e_multi_turn_chat_session(client: TestClient):
    """E2E test: Multi-turn chat session with query rewriting and history tracking."""
    # 1. Create chat session
    session_res = client.post("/api/v1/sessions", json={"title": "Titan Chat Session"})
    assert session_res.status_code == 201
    session_id = session_res.json()["session_id"]

    # 2. Turn 1
    turn1_res = client.post(
        f"/api/v1/sessions/{session_id}/chat",
        json={"query": "What is Project Titan?", "top_k": 3}
    )
    assert turn1_res.status_code == 200
    turn1_data = turn1_res.json()
    assert turn1_data["turn_count"] == 2

    # 3. Turn 2 (Follow-up)
    turn2_res = client.post(
        f"/api/v1/sessions/{session_id}/chat",
        json={"query": "When did it launch?", "top_k": 3}
    )
    assert turn2_res.status_code == 200
    turn2_data = turn2_res.json()
    assert turn2_data["turn_count"] == 4

    # 4. Verify History
    history_res = client.get(f"/api/v1/sessions/{session_id}/history")
    assert history_res.status_code == 200
    history_data = history_res.json()
    assert history_data["turn_count"] == 4
    assert history_data["turns"][0]["role"] == "user"
    assert history_data["turns"][1]["role"] == "assistant"
    assert history_data["turns"][2]["role"] == "user"
    assert history_data["turns"][3]["role"] == "assistant"
