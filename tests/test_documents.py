import io
import pytest
from pypdf import PdfWriter
from backend.app.database.document_store import document_store


@pytest.fixture(autouse=True)
def clear_store():
    """Ensure document store is clean before each test."""
    document_store.clear()
    yield
    document_store.clear()


def create_sample_pdf_bytes() -> bytes:
    """Creates sample PDF bytes in memory for testing."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_upload_text_document(client):
    content = b"Header: Document Title\nPage 1 content text for RAG testing."
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("sample.txt", content, "text/plain")}
    )

    assert response.status_code == 201
    data = response.json()
    assert "document_id" in data
    assert data["filename"] == "sample.txt"
    assert data["total_pages"] == 1
    assert data["file_size"] == len(content)

    doc_id = data["document_id"]

    # Verify detail retrieval
    detail_res = client.get(f"/api/v1/documents/{doc_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["metadata"]["document_id"] == doc_id
    assert detail["metadata"]["filename"] == "sample.txt"
    assert len(detail["pages"]) == 1
    assert detail["pages"][0]["page_number"] == 1
    assert "Header: Document Title" in detail["pages"][0]["text"]


def test_upload_pdf_document(client):
    pdf_bytes = create_sample_pdf_bytes()
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("test_doc.pdf", pdf_bytes, "application/pdf")}
    )

    assert response.status_code == 201
    data = response.json()
    assert "document_id" in data
    assert data["filename"] == "test_doc.pdf"
    assert data["total_pages"] == 1

    doc_id = data["document_id"]

    # Check list endpoint
    list_res = client.get("/api/v1/documents/")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] == 1
    assert list_data["documents"][0]["document_id"] == doc_id

    # Delete document
    del_res = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200

    # Verify 404 after deletion
    get_res = client.get(f"/api/v1/documents/{doc_id}")
    assert get_res.status_code == 404


def test_invalid_file_extension(client):
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("script.exe", b"binary data", "application/octet-stream")}
    )
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


def test_empty_file_upload(client):
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
