import os
import json
import pytest
from pathlib import Path
from pypdf import PdfWriter
import io
from fastapi import HTTPException
from backend.app.services.ingestion.extractor import DocumentExtractor
from backend.app.services.security.upload_validator import AdvancedDocumentValidator


@pytest.mark.e2e
def test_manifest_and_fixture_loading():
    """Verify that manifest.json exists and contains expected structure."""
    manifest_path = Path("tests/fixtures/manifest.json")
    assert manifest_path.exists(), "manifest.json must exist in tests/fixtures/"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["version"] == "1.0"
    assert len(manifest["documents"]) >= 3
    assert len(manifest["unanswerable_questions"]) >= 1
    assert len(manifest["security_edge_cases"]) >= 1


@pytest.mark.e2e
def test_fixture_generation_and_extraction(tmp_path):
    """Generate dynamic fixtures matching manifest specs and verify DocumentExtractor."""
    manifest_path = Path("tests/fixtures/manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()

    # 1. Generate PDF fixture (sample_doc1.pdf)
    pdf_path = fixtures_dir / "sample_doc1.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)

    pdf_bytes_io = io.BytesIO()
    writer.write(pdf_bytes_io)
    pdf_bytes = pdf_bytes_io.getvalue()

    with open(pdf_path, "wb") as pf:
        pf.write(pdf_bytes)

    # Test extractor on text/markdown files defined in manifest
    for doc_def in manifest["documents"]:
        if doc_def["type"] in ("txt", "md"):
            doc_path = fixtures_dir / doc_def["filename"]
            content_lines = []
            for page in doc_def["pages"]:
                for fact in page["facts"]:
                    content_lines.append(fact["text"])
            file_content = "\n\n".join(content_lines).encode("utf-8")
            with open(doc_path, "wb") as df:
                df.write(file_content)

            pages = DocumentExtractor.extract(file_content, f".{doc_def['type']}")
            assert len(pages) >= 1
            extracted_text = pages[0].text
            for page in doc_def["pages"]:
                for fact in page["facts"]:
                    assert fact["text"] in extracted_text


@pytest.mark.e2e
def test_security_edge_case_fixtures(tmp_path):
    """Test validation and extraction rejection for edge cases (corrupt, empty, wrong magic bytes)."""
    # 1. Empty file -> raises HTTPException (status 400)
    with pytest.raises(HTTPException) as exc_info:
        AdvancedDocumentValidator.sniff_content_type_and_extension(b"", "empty.txt")
    assert exc_info.value.status_code == 400

    # 2. Corrupt PDF -> raises HTTPException (status 422) in validate_pdf_sanity
    corrupt_pdf = b"%PDF-1.4\ncorrupted stream data without trailer"
    with pytest.raises(HTTPException) as exc_info:
        AdvancedDocumentValidator.validate_pdf_sanity(corrupt_pdf)
    assert exc_info.value.status_code == 422

    # 3. Wrong magic bytes / executable header -> raises HTTPException (status 415)
    bad_bytes = b"MZ\x90\x00\x03\x00\x00\x00"
    with pytest.raises(HTTPException) as exc_info:
        AdvancedDocumentValidator.sniff_content_type_and_extension(bad_bytes, "fake_pdf.pdf")
    assert exc_info.value.status_code == 415
