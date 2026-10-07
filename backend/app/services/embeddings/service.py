from typing import List, Optional
from fastapi import HTTPException, status
from backend.app.models.embedding import ChunkEmbedding
from backend.app.services.embeddings.factory import get_embedding_provider
from backend.app.services.embeddings.base import BaseEmbeddingProvider
from backend.app.database.chunk_store import chunk_store, ChunkStore


class EmbeddingService:
    """Modular embedding service for generating dense vector representations."""

    def __init__(self, provider: Optional[BaseEmbeddingProvider] = None, chk_store: ChunkStore = chunk_store):
        self._provider = provider
        self.chk_store = chk_store

    @property
    def provider(self) -> BaseEmbeddingProvider:
        if self._provider is None:
            self._provider = get_embedding_provider()
        return self._provider

    def set_provider(self, provider: BaseEmbeddingProvider):
        """Dynamically replace the active embedding provider at runtime."""
        self._provider = provider

    def embed_text(self, text: str) -> List[float]:
        """Generates a dense vector embedding for a single string query."""
        if not text or not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input text for embedding cannot be empty."
            )
        return self.provider.embed_query(text)

    def embed_document_chunks(self, document_id: str) -> List[ChunkEmbedding]:
        """
        Retrieves generated chunks for a document from chunk store,
        generates dense vector embeddings for each chunk, and returns ChunkEmbedding objects.
        """
        chunks = self.chk_store.get_chunks_by_document(document_id)
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No text chunks found for document ID '{document_id}'. Process chunks first."
            )

        texts = [c.text for c in chunks]
        vectors = self.provider.embed_documents(texts)

        results: List[ChunkEmbedding] = []
        for chunk_obj, vec in zip(chunks, vectors):
            results.append(
                ChunkEmbedding(
                    chunk=chunk_obj,
                    embedding=vec,
                    embedding_model=self.provider.model_name,
                    dimension=self.provider.dimension
                )
            )

        return results


embedding_service = EmbeddingService()
