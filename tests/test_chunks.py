import pytest
from backend.app.models.chunk import ChunkingConfig
from backend.app.services.chunking.splitter import TextSplitter
from backend.app.database.document_store import document_store
from backend.app.database.chunk_store import chunk_store


@pytest.fixture(autouse=True)
def clear_stores():
    """Clean stores before each test."""
    document_store.clear()
    chunk_store.clear()
    yield
    document_store.clear()
    chunk_store.clear()


def test_text_splitter_basic():
    config = ChunkingConfig(chunk_size=50, chunk_overlap=10)
    splitter = TextSplitter(config)
    sample_text = (
        "Retrieval-Augmented Generation (RAG) is an AI framework. "
        "It combines dense retrieval with LLM generation to produce grounded answers."
    )
    chunks = splitter.split_text(sample_text)
    assert len(chunks) > 1
    for chunk_str, start, end in chunks:
        assert len(chunk_str) <= 50


def test_text_splitter_invalid_config():
    with pytest.raises(ValueError, match="chunk_overlap must be strictly less than chunk_size"):
        TextSplitter(ChunkingConfig(chunk_size=50, chunk_overlap=50))


def test_chunk_processing_api_workflow(client):
    # 1. Upload a document first
    content = b"Paragraph 1 text for RAG.\n\nParagraph 2 detailed information about BM25 search.\n\nParagraph 3 cross-encoder reranking details."
    upload_res = client.post(
        "/api/v1/documents/upload",
        files={"file": ("rag_paper.txt", content, "text/plain")}
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]

    # 2. Process chunks via POST API
    chunk_res = client.post(
        f"/api/v1/chunks/process/{doc_id}?chunk_size=60&chunk_overlap=10"
    )
    assert chunk_res.status_code == 200
    chunk_data = chunk_res.json()
    assert chunk_data["document_id"] == doc_id
    assert chunk_data["total_chunks"] >= 3
    assert len(chunk_data["chunks"]) == chunk_data["total_chunks"]

    first_chunk = chunk_data["chunks"][0]
    assert first_chunk["document_id"] == doc_id
    assert first_chunk["filename"] == "rag_paper.txt"
    assert first_chunk["page_number"] == 1
    assert first_chunk["chunk_index"] == 0
    assert "source" in first_chunk

    # 3. Retrieve chunks via GET API
    get_res = client.get(f"/api/v1/chunks/{doc_id}")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["total_chunks"] == chunk_data["total_chunks"]

    # 4. Delete chunks via DELETE API
    del_res = client.delete(f"/api/v1/chunks/{doc_id}")
    assert del_res.status_code == 200

    # 5. Confirm deletion
    get_after_del = client.get(f"/api/v1/chunks/{doc_id}")
    assert get_after_del.json()["total_chunks"] == 0


def test_chunk_process_nonexistent_doc(client):
    response = client.post("/api/v1/chunks/process/nonexistent-uuid")
    assert response.status_code == 404
