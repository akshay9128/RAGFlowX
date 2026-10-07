import uuid
from typing import List
from fastapi import HTTPException, status
from backend.app.models.chunk import DocumentChunk, ChunkMetadata, ChunkingConfig
from backend.app.services.chunking.splitter import TextSplitter
from backend.app.database.document_store import document_store, DocumentStore
from backend.app.database.chunk_store import chunk_store, ChunkStore


class ChunkingService:
    """Orchestrates document chunking and metadata preservation."""

    def __init__(self, doc_store: DocumentStore = document_store, chk_store: ChunkStore = chunk_store):
        self.doc_store = doc_store
        self.chk_store = chk_store

    def process_document_chunks(
        self,
        document_id: str,
        config: ChunkingConfig = None
    ) -> List[DocumentChunk]:
        """
        Retrieves document pages from document store, splits pages into text chunks,
        attaches metadata lineage to each chunk, and saves them into chunk store.
        """
        doc = self.doc_store.get_document(document_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID '{document_id}' not found."
            )

        config = config or ChunkingConfig()
        splitter = TextSplitter(config)

        generated_chunks: List[DocumentChunk] = []
        global_chunk_index = 0

        meta = doc.metadata

        for page in doc.pages:
            splits = splitter.split_text(page.text)
            for text_piece, start_idx, end_idx in splits:
                chunk_id = str(uuid.uuid4())
                chunk_metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=meta.document_id,
                    filename=meta.filename,
                    page_number=page.page_number,
                    chunk_index=global_chunk_index,
                    source=meta.source,
                    character_count=len(text_piece),
                    start_char_idx=start_idx,
                    end_char_idx=end_idx
                )
                chunk_obj = DocumentChunk(text=text_piece, metadata=chunk_metadata)
                generated_chunks.append(chunk_obj)
                global_chunk_index += 1

        self.chk_store.save_chunks(document_id, generated_chunks)
        return generated_chunks


chunking_service = ChunkingService()
