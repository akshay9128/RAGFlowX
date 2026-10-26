from fastapi import APIRouter, UploadFile, File, Query, Depends, status
from backend.app.services.ingestion.service import DocumentIngestionService
from backend.app.services.chunking.service import ChunkingService
from backend.app.services.embeddings.service import EmbeddingService
from backend.app.services.retrieval.vector_service import VectorSearchService
from backend.app.services.retrieval.bm25_service import BM25SearchService
from backend.app.services.pipeline.service import PipelineService
from backend.app.database.document_store import DocumentStore
from backend.app.database.chunk_store import ChunkStore
from backend.app.database.vector_store.factory import get_vector_store

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

def get_pipeline_service() -> PipelineService:
    doc_store = DocumentStore()
    chunk_store = ChunkStore()
    ingestion_svc = DocumentIngestionService(doc_store)
    chunk_svc = ChunkingService(doc_store, chunk_store)
    embedding_svc = EmbeddingService()
    vector_store = get_vector_store()
    vector_svc = VectorSearchService(embedding_svc, vector_store)
    bm25_svc = BM25SearchService(chunk_store)
    return PipelineService(ingestion_svc, chunk_svc, embedding_svc, vector_svc, bm25_svc)

@router.post("/ingest", status_code=status.HTTP_201_CREATED)
async def ingest_pipeline_endpoint(
    file: UploadFile = File(...),
    chunk_size: int = Query(500, ge=100, le=5000),
    chunk_overlap: int = Query(50, ge=0, le=500),
    service: PipelineService = Depends(get_pipeline_service)
):
    """
    Single-step ingest pipeline: upload, extract, chunk, embed, vector-index, and BM25-index.
    """
    result = await service.ingest_document(file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return result
