import json
from pathlib import Path
from typing import Dict, List, Optional
from backend.app.config import settings
from backend.app.models.chunk import DocumentChunk


class ChunkStore:
    """Persistent storage manager for document text chunks."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._chunks_by_doc: Dict[str, List[DocumentChunk]] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        """Loads saved chunks from DATA_DIR."""
        for json_file in self.data_dir.glob("chunks_*.json"):
            try:
                doc_id = json_file.stem.replace("chunks_", "")
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    chunks = [DocumentChunk.model_validate(c) for c in data]
                    self._chunks_by_doc[doc_id] = chunks
            except Exception as exc:
                print(f"Warning: Failed to load chunks from {json_file}: {exc}")

    def save_chunks(self, document_id: str, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Saves chunks for a document in memory and to disk."""
        self._chunks_by_doc[document_id] = chunks
        file_path = self.data_dir / f"chunks_{document_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            data = [chunk.model_dump() for chunk in chunks]
            f.write(json.dumps(data, indent=2))
        return chunks

    def get_chunks_by_document(self, document_id: str) -> List[DocumentChunk]:
        """Retrieves all chunks for a specific document_id."""
        return self._chunks_by_doc.get(document_id, [])

    def get_all_chunks(self) -> List[DocumentChunk]:
        """Retrieves all chunks across all documents."""
        all_chunks: List[DocumentChunk] = []
        for chunks in self._chunks_by_doc.values():
            all_chunks.extend(chunks)
        return all_chunks

    def delete_chunks_by_document(self, document_id: str) -> bool:
        """Deletes all chunks for a document_id."""
        if document_id in self._chunks_by_doc:
            del self._chunks_by_doc[document_id]
            file_path = self.data_dir / f"chunks_{document_id}.json"
            if file_path.exists():
                file_path.unlink()
            return True
        return False

    def clear(self):
        """Clears all stored chunks (primarily for testing)."""
        self._chunks_by_doc.clear()
        for json_file in self.data_dir.glob("chunks_*.json"):
            try:
                json_file.unlink()
            except Exception:
                pass


# Global singleton instance
chunk_store = ChunkStore()
