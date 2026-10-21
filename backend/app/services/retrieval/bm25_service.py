import json
from typing import Dict, List, Optional
from rank_bm25 import BM25Okapi
import re

from backend.app.config import settings
from backend.app.database.chunk_store import chunk_store, ChunkStore
from backend.app.models.chunk import DocumentChunk
from backend.app.models.search import SearchResult


class BM25SearchService:
    """
    Service for BM25 keyword-based document retrieval.

    BM25 (Best Match 25) is a probabilistic ranking function used for
    keyword search. It excels at finding exact words, technical terms,
    and rare words that semantic vector search may miss.
    """

    def __init__(self, chunk_store: ChunkStore = chunk_store):
        self.chunk_store = chunk_store
        # In-memory BM25 index: document_id -> BM25 index
        self._bm25_indices: Dict[str, BM25Okapi] = {}
        # Tokenized chunks storage for re-indexing: document_id -> list of chunks
        self._indexed_chunks: Dict[str, List[DocumentChunk]] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        """Loads BM25 index data from persistent storage."""
        bm25_data_file = settings.DATA_DIR / "bm25_index.json"
        if bm25_data_file.exists():
            try:
                with open(bm25_data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Restore indexed chunks metadata
                    self._indexed_chunks = {
                        doc_id: [
                            DocumentChunk.model_validate(chunk_data)
                            for chunk_data in chunks_data
                        ]
                        for doc_id, chunks_data in data.get("indexed_chunks", {}).items()
                    }
                    # Rebuild BM25 indices from loaded chunks
                    for doc_id, chunks in self._indexed_chunks.items():
                        tokenized_chunks = [self._tokenize(chunk.text) for chunk in chunks]
                        self._bm25_indices[doc_id] = BM25Okapi(tokenized_chunks)
            except Exception as exc:
                print(f"Warning: Failed to load BM25 index from {bm25_data_file}: {exc}")

    def _save_to_disk(self):
        """Saves BM25 index metadata to persistent storage."""
        bm25_data_file = settings.DATA_DIR / "bm25_index.json"
        data = {
            "indexed_chunks": {
                doc_id: [chunk.model_dump() for chunk in chunks]
                for doc_id, chunks in self._indexed_chunks.items()
            }
        }
        with open(bm25_data_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2))

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text for BM25 indexing.

        Converts to lowercase, removes special characters, and splits
        on whitespace to produce token list.
        """
        # Convert to lowercase and remove special characters
        text = text.lower()
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Split on whitespace and filter empty strings
        tokens = [token for token in text.split() if token]
        return tokens

    def index_document(self, document_id: str) -> int:
        """
        Builds BM25 index for all chunks of a document.

        Returns the number of chunks indexed.
        """
        chunks = self.chunk_store.get_chunks_by_document(document_id)
        if not chunks:
            return 0

        # Store chunks for persistence and retrieval
        self._indexed_chunks[document_id] = chunks

        # Tokenize all chunk texts
        tokenized_chunks = [self._tokenize(chunk.text) for chunk in chunks]

        # Build BM25 index
        self._bm25_indices[document_id] = BM25Okapi(tokenized_chunks)

        # Save metadata to disk
        self._save_to_disk()

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Performs BM25 keyword search.

        Args:
            query: Search query string
            top_k: Number of top results to return
            document_id: Optional scope to single document

        Returns:
            List of SearchResult objects sorted by BM25 score
        """
        if not query or not query.strip():
            return []

        # Tokenize query
        tokenized_query = self._tokenize(query)

        results: List[tuple[DocumentChunk, float]] = []

        if document_id:
            # Search within single document
            if document_id in self._bm25_indices:
                bm25 = self._bm25_indices[document_id]
                chunk_list = self._indexed_chunks.get(document_id, [])

                scores = bm25.get_scores(tokenized_query)

                for idx, score in enumerate(scores):
                    if idx < len(chunk_list):
                        results.append((chunk_list[idx], float(score)))
        else:
            # Search across all indexed documents
            for doc_id, bm25 in self._bm25_indices.items():
                chunk_list = self._indexed_chunks.get(doc_id, [])
                if not chunk_list:
                    continue

                scores = bm25.get_scores(tokenized_query)

                for idx, score in enumerate(scores):
                    if idx < len(chunk_list):
                        results.append((chunk_list[idx], float(score)))

        # Filter out zero/negative scores before sorting
        results = [(chunk, score) for chunk, score in results if score > 0]

        # Sort by score descending and take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:top_k]

        # Convert to SearchResult objects
        search_results = []
        for chunk, score in top_results:
            # Normalize score to [0, 1] range (BM25 scores can be unbounded)
            # Using a sigmoid-like normalization for practical use
            normalized_score = min(max(score / 10.0, 0.0), 1.0)

            search_results.append(
                SearchResult(
                    chunk_id=chunk.metadata.chunk_id,
                    document_id=chunk.metadata.document_id,
                    filename=chunk.metadata.filename,
                    page_number=chunk.metadata.page_number,
                    chunk_index=chunk.metadata.chunk_index,
                    text=chunk.text,
                    score=normalized_score,
                    source=chunk.metadata.source
                )
            )

        return search_results

    def get_stats(self) -> dict:
        """Returns BM25 index statistics."""
        total_chunks = sum(len(chunks) for chunks in self._indexed_chunks.values())
        return {
            "provider": "bm25",
            "total_indexed_chunks": total_chunks,
            "indexed_documents": len(self._indexed_chunks)
        }

    def delete_document(self, document_id: str) -> bool:
        """Deletes BM25 index for a document."""
        if document_id in self._bm25_indices:
            del self._bm25_indices[document_id]

        if document_id in self._indexed_chunks:
            del self._indexed_chunks[document_id]
            self._save_to_disk()
            return True

        return False

    def clear(self):
        """Clears all BM25 indices."""
        self._bm25_indices.clear()
        self._indexed_chunks.clear()
        bm25_data_file = settings.DATA_DIR / "bm25_index.json"
        if bm25_data_file.exists():
            bm25_data_file.unlink()


# Global singleton instance
bm25_search_service = BM25SearchService()