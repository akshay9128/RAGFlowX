import uuid
from pathlib import Path
from fastapi import UploadFile
from backend.app.config import settings
from backend.app.models.document import ProcessedDocument, DocumentMetadata
from backend.app.services.ingestion.validator import DocumentValidator
from backend.app.services.ingestion.extractor import DocumentExtractor
from backend.app.database.document_store import document_store, DocumentStore


class DocumentIngestionService:
    """Orchestrates document validation, storage, text extraction, and metadata preservation."""

    def __init__(self, store: DocumentStore = document_store):
        self.store = store

    async def process_and_ingest(self, file: UploadFile) -> ProcessedDocument:
        """Processes an uploaded file, extracts text pages and metadata, and persists the document."""
        content = await file.read()

        # 1. Validate file
        _, file_extension = DocumentValidator.validate_file(file, content)

        # 2. Generate unique document_id
        doc_id = str(uuid.uuid4())
        safe_filename = file.filename or "uploaded_document"
        saved_path = settings.UPLOAD_DIR / f"{doc_id}_{safe_filename}"

        # 3. Save raw file to disk
        with open(saved_path, "wb") as f:
            f.write(content)

        # 4. Extract pages with metadata
        pages = DocumentExtractor.extract(content, file_extension)

        # 5. Build metadata preservation
        metadata = DocumentMetadata(
            document_id=doc_id,
            filename=safe_filename,
            content_type=file.content_type or "application/octet-stream",
            file_size=len(content),
            total_pages=len(pages),
            source=str(saved_path)
        )

        processed_doc = ProcessedDocument(
            metadata=metadata,
            pages=pages
        )

        # 6. Store processed document
        self.store.save_document(processed_doc)

        return processed_doc


ingestion_service = DocumentIngestionService()
