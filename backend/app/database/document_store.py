import json
from pathlib import Path
from typing import Dict, List, Optional
from backend.app.config import settings
from backend.app.models.document import ProcessedDocument, DocumentMetadata


class DocumentStore:
    """Persistent storage manager for processed documents and metadata."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._documents: Dict[str, ProcessedDocument] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        """Loads saved document metadata and pages from DATA_DIR."""
        for json_file in self.data_dir.glob("doc_*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    doc = ProcessedDocument.model_validate(data)
                    self._documents[doc.metadata.document_id] = doc
            except Exception as exc:
                print(f"Warning: Failed to load document from {json_file}: {exc}")

    def save_document(self, doc: ProcessedDocument) -> ProcessedDocument:
        """Saves a processed document in-memory and to disk."""
        self._documents[doc.metadata.document_id] = doc
        file_path = self.data_dir / f"doc_{doc.metadata.document_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc.model_dump_json(indent=2))
        return doc

    def get_document(self, document_id: str) -> Optional[ProcessedDocument]:
        """Retrieves a document by document_id."""
        return self._documents.get(document_id)

    def list_documents(self) -> List[DocumentMetadata]:
        """Lists metadata for all stored documents."""
        return [doc.metadata for doc in self._documents.values()]

    def delete_document(self, document_id: str) -> bool:
        """Deletes a document by document_id."""
        if document_id in self._documents:
            del self._documents[document_id]
            file_path = self.data_dir / f"doc_{document_id}.json"
            if file_path.exists():
                file_path.unlink()
            return True
        return False

    def clear(self):
        """Clears all stored documents (primarily for testing)."""
        self._documents.clear()
        for json_file in self.data_dir.glob("doc_*.json"):
            try:
                json_file.unlink()
            except Exception:
                pass


# Global singleton instance
document_store = DocumentStore()
