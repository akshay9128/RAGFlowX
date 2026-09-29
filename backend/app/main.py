from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.api.router import api_router

app = FastAPI(
    title=settings.APP_NAME,
    description="Hybrid RAG System backend API with Dense Search, BM25, Reranking, and Verified Citations",
    version="1.0.0",
    debug=settings.DEBUG
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include main API router with version prefix
app.include_router(api_router, prefix=settings.API_PREFIX)


# Also expose root health endpoint
@app.get("/health", tags=["Health"])
async def root_health():
    return {
        "status": "ok",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "version": "1.0.0"
    }


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME} API",
        "docs": "/docs",
        "health": "/health"
    }
