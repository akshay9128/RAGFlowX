import io
from typing import List
from pypdf import PdfReader
from fastapi import HTTPException, status
from backend.app.models.document import DocumentPage


class DocumentExtractor:
    """Extracts text and page-level metadata from documents (PDF, TXT, MD)."""

    @staticmethod
    def extract_from_pdf(content: bytes) -> List[DocumentPage]:
        """Extracts text page by page from a PDF file using pypdf."""
        try:
            pdf_file = io.BytesIO(content)
            reader = PdfReader(pdf_file)
            pages: List[DocumentPage] = []

            for page_idx, page in enumerate(reader.pages, start=1):
                extracted_text = page.extract_text() or ""
                # Clean up null bytes or unwanted control characters
                cleaned_text = extracted_text.replace("\x00", "").strip()

                pages.append(
                    DocumentPage(
                        page_number=page_idx,
                        text=cleaned_text,
                        character_count=len(cleaned_text)
                    )
                )

            if not pages:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Could not extract any pages from the PDF document."
                )

            return pages
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to process PDF document: {str(exc)}"
            )

    @staticmethod
    def extract_from_text(content: bytes) -> List[DocumentPage]:
        """Extracts text from plain text or markdown file as a single-page document."""
        try:
            text = content.decode("utf-8", errors="replace").strip()
            return [
                DocumentPage(
                    page_number=1,
                    text=text,
                    character_count=len(text)
                )
            ]
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to decode text document: {str(exc)}"
            )

    @classmethod
    def extract(cls, content: bytes, file_extension: str) -> List[DocumentPage]:
        """Routing method to extract document pages based on extension."""
        ext = file_extension.lower()
        if ext == ".pdf":
            return cls.extract_from_pdf(content)
        elif ext in {".txt", ".md"}:
            return cls.extract_from_text(content)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported extension for extraction: {file_extension}"
            )
