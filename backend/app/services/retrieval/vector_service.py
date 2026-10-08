from typing import List, Optional
from fastapi import HTTPException, status
from backend.app.models.search import VectorSearchResult
from backend.app.services.embeddings.service import embedding_service, EmbeddingService
from backend.app.database.vector_store.factory import get_vector_store
from backend.app.database.vector_store.base import BaseVectorStore


class VectorSearchService:
    """Service orchestrating document vector indexing and dense similarity search."""

    def __init__(
        self,
        embed_service: EmbeddingService = embedding_service,
        vec_store: Optional[BaseVectorStore] = None
    ):
        self.embed_service = embed_service
        self._vec_store = vec_store

    @property
    def vector_store(self) -> BaseVectorStore:
        if self._vec_store is None:
            self._vec_store = get_vector_store()
        return self._vec_store

    def set_vector_store(self, vec_store: BaseVectorStore):
        """Dynamically replace the active vector store at runtime."""
        self._vec_store = vec_store

    def index_document_chunks(self, document_id: str) -> int:
        """
        Generates dense vector embeddings for all chunks of a document,
        indexes them into the vector database, and returns the count indexed.
        """
        chunk_embeddings = self.embed_service.embed_document_chunks(document_id)
        if not chunk_embeddings:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No chunk embeddings could be generated for document '{document_id}'."
            )
        indexed_count = self.vector_store.add_embeddings(chunk_embeddings)
        return indexed_count

    def search_similar_chunks(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None
    ) -> List[VectorSearchResult]:
        """
        Embeds the input query string into a dense vector, performs Cosine Similarity
        search against indexed document vectors, and returns top_k relevant chunks.
        """
        if not query or not query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Search query string cannot be empty."
            )

        query_vector = self.embed_service.embed_text(query)
        results = self.vector_store.similarity_search(
            query_vector=query_vector,
            top_k=top_k,
            document_id=document_id
        )
        return results

    def get_stats(self) -> dict:
        """Returns vector database statistics."""
        return {
            "provider": self.vector_store.provider_name,
            "total_vectors": self.vector_store.get_total_vector_count(),
            "dimension": self.embed_service.provider.dimension
        }

    def delete_document_vectors(self, document_id: str) -> bool:
        """Deletes all indexed vectors for a document ID."""
        return self.vector_store.delete_by_document(document_id)


vector_search_service = VectorSearchService()
