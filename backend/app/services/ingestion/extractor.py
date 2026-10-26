import io
import zipfile
import xml.etree.ElementTree as ET
from typing import List
from pypdf import PdfReader
from fastapi import HTTPException, status
from backend.app.models.document import DocumentPage


class DocumentExtractor:
    """Extracts text and page-level metadata from documents (PDF, DOCX, TXT, MD, ZIP, etc.)."""

    @staticmethod
    def extract_from_pdf(content: bytes) -> List[DocumentPage]:
        """Extracts text page by page from a PDF file using pypdf."""
        try:
            pdf_file = io.BytesIO(content)
            reader = PdfReader(pdf_file)
            pages: List[DocumentPage] = []

            for page_idx, page in enumerate(reader.pages, start=1):
                extracted_text = page.extract_text() or ""
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
    def extract_from_docx(content: bytes) -> List[DocumentPage]:
        """Extracts text from a DOCX file using built-in zipfile and xml.etree.ElementTree (namespace-agnostic)."""
        try:
            docx_file = io.BytesIO(content)
            with zipfile.ZipFile(docx_file) as z:
                if 'word/document.xml' not in z.namelist():
                    raise ValueError("Invalid docx structure: word/document.xml missing")
                xml_content = z.read('word/document.xml')
                tree = ET.fromstring(xml_content)

                paragraphs = []
                for elem in tree.iter():
                    tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                    if tag == 'p':
                        p_texts = []
                        for subelem in elem.iter():
                            subtag = subelem.tag.split('}')[-1] if '}' in subelem.tag else subelem.tag
                            if subtag in ('t', 'text') and subelem.text:
                                p_texts.append(subelem.text)
                        if p_texts:
                            paragraphs.append(''.join(p_texts))

                text = '\n\n'.join(paragraphs).strip()
                if not text:
                    all_texts = [elem.text for elem in tree.iter() if (elem.tag.split('}')[-1] in ('t', 'text')) and elem.text]
                    text = '\n'.join(all_texts).strip()

                if not text:
                    text = content.decode("utf-8", errors="replace").strip()

                return [
                    DocumentPage(
                        page_number=1,
                        text=text,
                        character_count=len(text)
                    )
                ]
        except Exception as exc:
            text = content.decode("utf-8", errors="replace").strip()
            return [
                DocumentPage(
                    page_number=1,
                    text=text,
                    character_count=len(text)
                )
            ]

    @staticmethod
    def extract_from_text(content: bytes) -> List[DocumentPage]:
        """Extracts text from plain text or markup file as a single-page document."""
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

    @staticmethod
    def extract_from_zip(content: bytes) -> List[DocumentPage]:
        """Extracts text from all supported documents inside a ZIP archive."""
        try:
            zip_file = io.BytesIO(content)
            all_pages: List[DocumentPage] = []
            page_offset = 1
            with zipfile.ZipFile(zip_file) as z:
                for filename in z.namelist():
                    if filename.endswith('/') or filename.startswith('__MACOSX'):
                        continue
                    file_bytes = z.read(filename)
                    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
                    try:
                        if ext == ".pdf":
                            pages = DocumentExtractor.extract_from_pdf(file_bytes)
                        elif ext == ".docx":
                            pages = DocumentExtractor.extract_from_docx(file_bytes)
                        elif ext in {".txt", ".md", ".csv", ".json", ".html", ".rtf"}:
                            pages = DocumentExtractor.extract_from_text(file_bytes)
                        else:
                            continue

                        for p in pages:
                            all_pages.append(
                                DocumentPage(
                                    page_number=page_offset,
                                    text=f"[File: {filename} - Page {p.page_number}]\n{p.text}",
                                    character_count=len(p.text)
                                )
                            )
                            page_offset += 1
                    except Exception:
                        continue
            if not all_pages:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="No valid extractable documents found inside the ZIP archive."
                )
            return all_pages
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to process ZIP archive: {str(exc)}"
            )

    @classmethod
    def extract(cls, content: bytes, file_extension: str) -> List[DocumentPage]:
        """Routing method to extract document pages based on extension."""
        ext = file_extension.lower()
        if ext == ".pdf":
            return cls.extract_from_pdf(content)
        elif ext == ".docx":
            return cls.extract_from_docx(content)
        elif ext == ".zip":
            return cls.extract_from_zip(content)
        elif ext in {".txt", ".md", ".csv", ".json", ".html", ".rtf", ".doc", ".xlsx", ".pptx"}:
            return cls.extract_from_text(content)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported extension for extraction: {file_extension}"
            )
