from fastapi import APIRouter, status
from backend.app.schemas.rag import RAGQueryRequest, RAGQueryResponse
from backend.app.services.retrieval.rag_service import rag_service
from backend.app.config import settings

router = APIRouter(prefix="/rag", tags=["Basic RAG"])


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question over ingested documents (Basic RAG)"
)
async def query_rag_pipeline(payload: RAGQueryRequest):
    """
    Basic RAG Endpoint: Performs dense vector retrieval over uploaded documents,
    builds a grounded context prompt, and generates an answer using the LLM.
    """
    response = rag_service.query_rag(
        query=payload.query,
        top_k=payload.top_k or settings.RAG_TOP_K,
        document_id=payload.document_id
    )
    return response


@router.get(
    "/info",
    status_code=status.HTTP_200_OK,
    summary="Get RAG pipeline configuration and LLM model info"
)
async def get_rag_info():
    """Returns active LLM model, provider, vector store provider, and RAG configuration."""
    provider = rag_service.llm_provider
    return {
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": provider.model_name,
        "vector_store_provider": settings.VECTOR_STORE_PROVIDER,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "default_top_k": settings.RAG_TOP_K
    }
