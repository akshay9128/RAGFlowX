from fastapi import APIRouter
from backend.app.api.routes import health, documents, chunks

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(chunks.router)


