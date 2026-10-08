import pytest
from backend.app.services.embeddings.mock import MockEmbeddingProvider
from backend.app.services.embeddings.service import embedding_service
from backend.app.services.generation.mock import MockLLMProvider
from backend.app.services.retrieval.rag_service import rag_service
from backend.app.database.document_store import document_store
from backend.app.database.chunk_store import chunk_store
from backend.app.services.retrieval.vector_service import vector_search_service


@pytest.fixture(autouse=True)
def setup_mock_environment():
    """Ensure clean stores and mock embedding/LLM providers for fast end-to-end tests."""
    document_store.clear()
    chunk_store.clear()
    vector_search_service.vector_store.clear()

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


def test_rag_info_api(client):
    res = client.get("/api/v1/rag/info")
    assert res.status_code == 200
    data = res.json()
    assert "llm_provider" in data
    assert "llm_model" in data
    assert "vector_store_provider" in data


def test_rag_query_empty_store(client):
    res = client.post(
        "/api/v1/rag/query",
        json={"query": "What is the capital of France?", "top_k": 3}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["total_retrieved_chunks"] == 0
    assert "could not find enough relevant information" in data["answer"].lower()


def test_basic_rag_full_end_to_end(client):
    # 1. Upload document (Phase 2)
    doc_text = b"Retrieval Augmented Generation (RAG) grounds LLM responses using domain documents. The annual subscription fee for Acme RAG is $99 per year."
    upload_res = client.post(
        "/api/v1/documents/upload",
        files={"file": ("acme_pricing.txt", doc_text, "text/plain")}
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]

    # 2. Process text chunks (Phase 3)
    chunk_res = client.post(f"/api/v1/chunks/process/{doc_id}?chunk_size=120&chunk_overlap=20")
    assert chunk_res.status_code == 200

    # 3. Index embeddings into Vector Database (Phase 4 & 5)
    index_res = client.post(f"/api/v1/vector-store/index/{doc_id}")
    assert index_res.status_code == 200

    # 4. Perform Basic RAG Query (Phase 6)
    rag_res = client.post(
        "/api/v1/rag/query",
        json={
            "query": "How much does Acme RAG cost per year?",
            "top_k": 3
        }
    )
    assert rag_res.status_code == 200
    rag_data = rag_res.json()
    assert rag_data["query"] == "How much does Acme RAG cost per year?"
    assert rag_data["total_retrieved_chunks"] >= 1
    assert len(rag_data["retrieved_chunks"]) == rag_data["total_retrieved_chunks"]
    assert "answer" in rag_data
    assert len(rag_data["answer"]) > 0

    top_chunk = rag_data["retrieved_chunks"][0]
    assert top_chunk["document_id"] == doc_id
    assert top_chunk["filename"] == "acme_pricing.txt"
    assert "score" in top_chunk
