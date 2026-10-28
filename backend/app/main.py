import uuid
from fastapi import FastAPI, Request, Response, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from backend.app.config import settings
from backend.app.api.router import api_router
from backend.app.middleware.error_handler import request_tracing_middleware, register_exception_handlers
from backend.app.core.redaction import setup_log_redaction
from backend.app.core.rate_limiter import check_rate_limit
from backend.app.core.security import verify_api_key_dependency

# Initialize log redaction
setup_log_redaction()

app = FastAPI(
    title=settings.APP_NAME,
    description="Hybrid RAG System backend API with Dense Search, BM25, Reranking, and Verified Citations",
    version="1.0.0",
    debug=settings.DEBUG
)

# Set up CORS middleware with configured origins
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",")] if settings.CORS_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register request tracing & error handling middleware
app.middleware("http")(request_tracing_middleware)


@app.middleware("http")
async def security_headers_and_rate_limit_middleware(request: Request, call_next):
    # Rate limit check (may raise HTTPException(429))
    response = Response()
    try:
        await check_rate_limit(request, response)
    except HTTPException as exc:
        rate_headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower().startswith("x-ratelimit-") or k.lower() == "retry-after"
        }
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        res = JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "error": {
                    "code": "TOO_MANY_REQUESTS",
                    "message": exc.detail,
                    "details": {},
                    "request_id": request_id
                }
            },
            headers=rate_headers,
        )
        if settings.SECURITY_HEADERS_ENABLED:
            res.headers["X-Request-ID"] = request_id
            res.headers["X-Content-Type-Options"] = "nosniff"
            res.headers["X-Frame-Options"] = "DENY"
            res.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            res.headers["Content-Security-Policy"] = "default-src 'self' 'unsafe-inline' 'unsafe-eval' cdn.jsdelivr.net cdnjs.cloudflare.com unpkg.com;"
        return res

    res = await call_next(request)

    # Copy rate limit headers if present
    for k, v in response.headers.items():
        if k.lower().startswith("x-ratelimit-") or k.lower() == "retry-after":
            res.headers[k] = v

    # Security headers
    if settings.SECURITY_HEADERS_ENABLED:
        res.headers["X-Content-Type-Options"] = "nosniff"
        res.headers["X-Frame-Options"] = "DENY"
        res.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        res.headers["Content-Security-Policy"] = "default-src 'self' 'unsafe-inline' 'unsafe-eval' cdn.jsdelivr.net cdnjs.cloudflare.com unpkg.com;"

    return res


# Register global exception handlers
register_exception_handlers(app)

# Include main API router with version prefix and global API key auth dependency.
# verify_api_key_dependency checks settings.AUTH_ENABLED per-request, so it is
# always registered and no-ops when auth is disabled.
app.include_router(
    api_router,
    prefix=settings.API_PREFIX,
    dependencies=[Depends(verify_api_key_dependency)]
)

# Mount static frontend files
app.mount("/static", StaticFiles(directory="frontend"), name="static")


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
        "health": "/health",
        "ui": "/ui"
    }


@app.get("/ui", tags=["UI"])
async def serve_frontend():
    """Serves the RAGFlowX web UI interface."""
    return FileResponse("frontend/index.html")
