import pytest
from backend.app.services.embeddings.mock import MockEmbeddingProvider
from backend.app.services.embeddings.factory import get_embedding_provider
from backend.app.services.embeddings.service import embedding_service
from backend.app.database.document_store import document_store
from backend.app.database.chunk_store import chunk_store


@pytest.fixture(autouse=True)
def use_mock_embedding_provider():
    """Ensure fast deterministic mock embedding provider is active for unit tests."""
    document_store.clear()
    chunk_store.clear()
    original_provider = embedding_service.provider
    mock_prov = MockEmbeddingProvider(dim=384)
    embedding_service.set_provider(mock_prov)
    yield
    embedding_service.set_provider(original_provider)
    document_store.clear()
    chunk_store.clear()


def test_mock_embedding_provider_dimensions():
    provider = MockEmbeddingProvider(dim=384)
    vec = provider.embed_query("Hello RAG world")
    assert len(vec) == 384
    # Ensure vector elements are floats
    assert isinstance(vec[0], float)

    docs_vecs = provider.embed_documents(["Text one", "Text two"])
    assert len(docs_vecs) == 2
    assert len(docs_vecs[0]) == 384


def test_embedding_factory():
    mock_prov = get_embedding_provider(provider_type="mock")
    assert mock_prov.dimension == 384
    assert "mock" in mock_prov.model_name.lower()


def test_embedding_info_api(client):
    res = client.get("/api/v1/embeddings/info")
    assert res.status_code == 200
    data = res.json()
    assert "provider" in data
    assert "dimension" in data
    assert "model_name" in data


def test_embed_single_text_api(client):
    res = client.post(
        "/api/v1/embeddings/text",
        json={"text": "What is dense retrieval?"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["text"] == "What is dense retrieval?"
    assert data["dimension"] == 384
    assert len(data["embedding"]) == 384


def test_embed_document_chunks_api_workflow(client):
    # 1. Upload document
    content = b"Dense search maps text to high dimensional vectors for similarity search."
    upload_res = client.post(
        "/api/v1/documents/upload",
        files={"file": ("dense.txt", content, "text/plain")}
    )
    doc_id = upload_res.json()["document_id"]

    # 2. Process chunks
    client.post(f"/api/v1/chunks/process/{doc_id}?chunk_size=100&chunk_overlap=10")

    # 3. Generate chunk embeddings
    embed_res = client.post(f"/api/v1/embeddings/chunks/{doc_id}")
    assert embed_res.status_code == 200
    data = embed_res.json()
    assert data["document_id"] == doc_id
    assert data["total_embeddings"] >= 1
    assert data["dimension"] == 384
    assert len(data["embeddings"][0]["embedding"]) == 384
    assert data["embeddings"][0]["document_id"] == doc_id
