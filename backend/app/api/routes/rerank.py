from fastapi import APIRouter
from backend.app.schemas.rerank import RerankRequest, RerankResponse, RerankResultItem
from backend.app.services.reranking.service import rerank_service

router = APIRouter(prefix="/rerank", tags=["Cross-Encoder Reranking"])


@router.post(
    "/",
    response_model=RerankResponse,
    summary="Rerank document passages using Cross-Encoder"
)
async def rerank_documents(payload: RerankRequest):
    """
    Applies Cross-Encoder reranking to a list of text documents against a query.
    """
    provider = rerank_service.provider
    scored_pairs = provider.rerank(
        query=payload.query,
        documents=payload.documents,
        top_k=payload.top_k
    )

    items = [
        RerankResultItem(
            index=orig_idx,
            text=payload.documents[orig_idx],
            score=score
        )
        for orig_idx, score in scored_pairs
    ]

    return RerankResponse(
        query=payload.query,
        total_results=len(items),
        results=items,
        model_name=provider.model_name
    )
