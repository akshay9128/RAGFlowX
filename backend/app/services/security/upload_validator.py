import os
import re
import logging
from pathlib import Path
from typing import Tuple, Optional
from fastapi import UploadFile, HTTPException, status
from pypdf import PdfReader
from backend.app.config import settings

logger = logging.getLogger(__name__)


class AntivirusScanner:
    """Pluggable antivirus interface (no-op default implementation)."""
    def scan_bytes(self, content: bytes, filename: str) -> bool:
        if not settings.ANTIVIRUS_ENABLED:
            return True
        # Stub for ClamAV or external AV integration
        return True


antivirus_scanner = AntivirusScanner()


class AdvancedDocumentValidator:
    """Hardened document validator handling magic bytes, PDF sanity, filename sanitization, and streaming size limits."""

    RESERVED_NAMES = {"con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9"}

    @classmethod
    def sanitize_filename(cls, filename: Optional[str]) -> str:
        if not filename:
            return "uploaded_document.txt"

        # Strip path separators (/ and \)
        base = os.path.basename(filename.replace("\\", "/"))
        # Remove null bytes and control characters (keep ord >= 32, drop DEL=127)
        base = "".join(ch for ch in base if ord(ch) >= 32 and ord(ch) != 127)
        # Strip leading/trailing dots and spaces
        base = base.strip(". ")
        if not base:
            base = "uploaded_document.txt"

        # Check reserved Windows names
        stem = Path(base).stem.lower()
        if stem in cls.RESERVED_NAMES:
            base = f"file_{base}"

        # Cap filename length to 255 chars
        if len(base) > 255:
            ext = Path(base).suffix
            base = Path(base).stem[:255 - len(ext)] + ext

        return base

    @classmethod
    def sniff_content_type_and_extension(cls, content: bytes, filename: str) -> Tuple[str, str]:
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )

        ext = Path(filename).suffix.lower()

        # Check magic bytes
        if content.startswith(b"%PDF-"):
            detected_type = "application/pdf"
            if ext != ".pdf":
                ext = ".pdf"
        else:
            # Check for binary executable signatures (reject outright)
            if content.startswith(b"MZ") or content.startswith(b"\x7fELF") or b"\x00\x00\x00\x00" in content[:1024]:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="File content does not match allowed document types (Binary/Executable signature detected)."
                )

            # Try decoding as text / markdown
            try:
                content.decode("utf-8")
                if ext in (".md", ".markdown"):
                    detected_type = "text/markdown"
                    ext = ".md"
                else:
                    detected_type = "text/plain"
                    ext = ".txt"
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="File content is not valid UTF-8 text or supported PDF format."
                )

        if ext not in (".pdf", ".txt", ".md", ".markdown"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file extension '{ext}'. Allowed: .pdf, .txt, .md"
            )

        return detected_type, ext

    @classmethod
    def validate_pdf_sanity(cls, content: bytes) -> dict:
        """Inspects PDF structure for encryption, page counts, and risky features (JS, Launch actions)."""
        import io
        try:
            reader = PdfReader(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Corrupted or invalid PDF structure: {str(e)}"
            )

        if reader.is_encrypted:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Encrypted PDFs are not supported."
            )

        page_count = len(reader.pages)
        if page_count > settings.UPLOAD_MAX_PAGE_COUNT:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"PDF page count ({page_count}) exceeds maximum allowed ({settings.UPLOAD_MAX_PAGE_COUNT})."
            )

        # Inspect for risky features (JS, Launch)
        risky_found = False
        try:
            # Check trailer / catalog dictionaries for /JS, /JavaScript, /Launch
            catalog = reader.trailer.get("/Root")
            if catalog:
                catalog_obj = catalog.get_object() if hasattr(catalog, "get_object") else catalog
                if isinstance(catalog_obj, dict):
                    if "/JS" in catalog_obj or "/JavaScript" in catalog_obj or "/Launch" in catalog_obj:
                        risky_found = True
        except Exception:
            pass

        if risky_found:
            if settings.PDF_RISKY_FEATURE_MODE == "reject":
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="PDF contains prohibited active/risky features (Embedded JavaScript or Launch actions)."
                )
            else:
                logger.warning("PDF contains risky features, but mode is set to flag only.")

        return {"page_count": page_count, "risky_features_detected": risky_found}

    @classmethod
    async def validate_and_read_stream(cls, file: UploadFile) -> Tuple[bytes, str, str]:
        """Reads file upload while enforcing streaming body size limits."""
        max_size = settings.UPLOAD_MAX_BODY_SIZE_BYTES
        chunks = []
        total_size = 0

        while True:
            chunk = await file.read(65536)  # 64KB chunks
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > max_size:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Uploaded file exceeds maximum allowed size ({max_size / (1024*1024):.1f}MB)."
                )
            chunks.append(chunk)

        content = b"".join(chunks)
        sanitized_filename = cls.sanitize_filename(file.filename)
        content_type, ext = cls.sniff_content_type_and_extension(content, sanitized_filename)

        if ext == ".pdf":
            cls.validate_pdf_sanity(content)

        if not antivirus_scanner.scan_bytes(content, sanitized_filename):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="File failed antivirus security inspection."
            )

        return content, sanitized_filename, ext
