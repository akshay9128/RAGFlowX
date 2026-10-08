from fastapi import APIRouter
from backend.app.api.routes import health, documents, chunks, embeddings, vector_store, rag

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(chunks.router)
api_router.include_router(embeddings.router)
api_router.include_router(vector_store.router)
api_router.include_router(rag.router)





