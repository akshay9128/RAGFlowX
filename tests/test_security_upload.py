import io
import pytest
from fastapi import UploadFile, HTTPException, status
from backend.app.services.security.upload_validator import AdvancedDocumentValidator


class MockUploadFile:
    """Minimal mock of Starlette UploadFile for testing."""
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content
        self.content_type = "application/octet-stream"
        self._read = False

    async def read(self, n: int = 65536) -> bytes:
        if self._read:
            return b""
        self._read = True
        return self._content


@pytest.fixture
def validator():
    return AdvancedDocumentValidator


def test_sanitize_filename_path_traversal(validator):
    """Path separators are stripped from filenames."""
    result = validator.sanitize_filename("../../etc/passwd")
    assert result == "etc_passwd" or "/" not in result and "\\" not in result
    assert ".." not in result


def test_sanitize_filename_reserved_name(validator):
    """Reserved Windows names get prefixed."""
    result = validator.sanitize_filename("con.txt")
    assert result.lower().startswith("file_con") or result == "file_con.txt"


def test_sanitize_filename_length_cap(validator):
    """Long filenames are capped at 255 chars."""
    long_name = "a" * 300 + ".txt"
    result = validator.sanitize_filename(long_name)
    assert len(result) <= 255


def test_sanitize_filename_null_and_control(validator):
    """Null bytes and control characters are removed."""
    result = validator.sanitize_filename("test\x00name.txt")
    assert "\x00" not in result


def test_sanitize_filename_empty(validator):
    """Empty or None filename gets a default."""
    assert validator.sanitize_filename("") == "uploaded_document.txt"
    assert validator.sanitize_filename(None) == "uploaded_document.txt"


def test_sniff_pdf_magic_bytes(validator):
    """PDF files are detected by magic bytes regardless of extension."""
    content = b"%PDF-1.4\nfake pdf content"
    detected_type, ext = validator.sniff_content_type_and_extension(content, "document.txt")
    assert detected_type == "application/pdf"
    assert ext == ".pdf"


def test_sniff_text_markdown(validator):
    """Text/markdown files are detected by UTF-8 decoding."""
    content = b"# Hello World\nThis is markdown."
    detected_type, ext = validator.sniff_content_type_and_extension(content, "doc.md")
    assert detected_type == "text/markdown"
    assert ext == ".md"


def test_sniff_text_plain(validator):
    """Plain text files without known extension default to .txt."""
    content = b"Just some text content."
    detected_type, ext = validator.sniff_content_type_and_extension(content, "doc")
    assert detected_type == "text/plain"
    assert ext == ".txt"


def test_sniff_empty_file_rejected(validator):
    """Empty file content is rejected."""
    with pytest.raises(HTTPException) as exc:
        validator.sniff_content_type_and_extension(b"", "empty.txt")
    assert exc.value.status_code == status.HTTP_400_BAD_REQUEST


def test_sniff_binary_executable_rejected(validator):
    """MZ/ELF executable signatures are rejected."""
    with pytest.raises(HTTPException) as exc:
        validator.sniff_content_type_and_extension(b"MZ\x90\x00fake exe", "program.pdf")
    assert exc.value.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


def test_sniff_invalid_utf8_rejected(validator):
    """Non-UTF-8 binary content is rejected."""
    with pytest.raises(HTTPException) as exc:
        validator.sniff_content_type_and_extension(b"\xff\xfe\x00\x00binary", "data.bin")
    assert exc.value.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


def test_validate_pdf_sanity_valid(validator):
    """A valid minimal PDF passes sanity checks."""
    # Minimal valid PDF structure
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n178\n%%EOF"
    )
    result = validator.validate_pdf_sanity(pdf_content)
    assert "page_count" in result
    assert result["risky_features_detected"] is False


def test_validate_pdf_encrypted_rejected(validator):
    """Encrypted PDFs are rejected."""
    pdf_content = b"%PDF-1.4\n1 0 obj\n<< /Encrypt << /V 1 /P -1024 >> >>\nendobj\n"
    with pytest.raises(HTTPException) as exc:
        validator.validate_pdf_sanity(pdf_content)
    assert exc.value.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_validate_and_read_stream_size_limit(validator):
    """Streaming read enforces max body size."""
    import asyncio

    async def run():
        big_content = b"x" * (25 * 1024 * 1024 + 1)  # Just over 25MB
        file = MockUploadFile("big.txt", big_content)
        with pytest.raises(HTTPException) as exc:
            await validator.validate_and_read_stream(file)
        assert exc.value.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE

    asyncio.run(run())
