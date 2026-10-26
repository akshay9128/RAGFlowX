import os
import uuid
import logging
from typing import Dict, Any, Optional
from fastapi import UploadFile
from backend.app.services.ingestion.service import DocumentIngestionService
from backend.app.services.chunking.service import ChunkingService
from backend.app.models.chunk import ChunkingConfig
from backend.app.services.embeddings.service import EmbeddingService
from backend.app.services.retrieval.vector_service import VectorSearchService
from backend.app.services.retrieval.bm25_service import BM25SearchService

logger = logging.getLogger(__name__)

class PipelineService:
    def __init__(
        self,
        ingestion_service: DocumentIngestionService,
        chunk_service: ChunkingService,
        embedding_service: EmbeddingService,
        vector_service: VectorSearchService,
        bm25_service: BM25SearchService
    ):
        self.ingestion_service = ingestion_service
        self.chunk_service = chunk_service
        self.embedding_service = embedding_service
        self.vector_service = vector_service
        self.bm25_service = bm25_service

    async def ingest_document(self, file: UploadFile, chunk_size: int = 500, chunk_overlap: int = 50) -> Dict[str, Any]:
        # 1. Upload & Extract
        processed_doc = await self.ingestion_service.process_and_ingest(file)
        document_id = processed_doc.metadata.document_id

        # 2. Chunk document
        config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = self.chunk_service.process_document_chunks(document_id, config=config)

        # 3. Embed & Index into Vector Store
        indexed_count = self.vector_service.index_document_chunks(document_id)

        # 4. Index into BM25
        bm25_count = self.bm25_service.index_document(document_id)

        return {
            "document_id": document_id,
            "filename": processed_doc.metadata.filename,
            "total_pages": processed_doc.metadata.total_pages,
            "total_chunks": len(chunks),
            "vector_indexed": indexed_count,
            "bm25_indexed": bm25_count,
            "status": "success"
        }
