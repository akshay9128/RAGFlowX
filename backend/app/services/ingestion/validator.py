from typing import Tuple
from fastapi import UploadFile, HTTPException, status

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


class DocumentValidator:
    """Validates uploaded document files before processing."""

    @staticmethod
    def validate_file(file: UploadFile, content: bytes) -> Tuple[bool, str]:
        """
        Validates an uploaded file's extension, size, and content.
        Raises HTTPException if invalid. Returns (is_valid, file_extension).
        """
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file must have a valid filename."
            )

        filename = file.filename.lower()
        extension = "." + filename.split(".")[-1] if "." in filename else ""

        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file extension '{extension}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes)."
            )

        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB."
            )

        return True, extension
