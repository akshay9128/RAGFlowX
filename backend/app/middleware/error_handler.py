import uuid
import time
import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from backend.app.config import settings

logger = logging.getLogger(__name__)

def add_security_headers(headers: dict):
    if settings.SECURITY_HEADERS_ENABLED:
        headers["X-Content-Type-Options"] = "nosniff"
        headers["X-Frame-Options"] = "DENY"
        headers["X-XSS-Protection"] = "1; mode=block"
        headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

async def request_tracing_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start_time = time.time()

    try:
        response = await call_next(request)
        duration = time.time() - start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration:.4f}"
        add_security_headers(response.headers)
        return response
    except Exception as exc:
        duration = time.time() - start_time
        logger.exception(f"Unhandled exception during request {request_id}: {exc}")
        headers = {"X-Request-ID": request_id}
        add_security_headers(headers)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": str(exc) if not isinstance(exc, StarletteHTTPException) else exc.detail,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": str(exc) if not isinstance(exc, StarletteHTTPException) else exc.detail,
                    "details": {},
                    "request_id": request_id
                }
            },
            headers=headers
        )

def register_exception_handlers(app):
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        headers = {"X-Request-ID": request_id}
        add_security_headers(headers)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": "Invalid request parameters or body",
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid request parameters or body",
                    "details": {"errors": exc.errors()},
                    "request_id": request_id
                }
            },
            headers=headers
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        code_map = {
            400: "BAD_REQUEST",
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            422: "UNPROCESSABLE_ENTITY",
            429: "TOO_MANY_REQUESTS",
        }
        error_code = code_map.get(exc.status_code, "HTTP_ERROR")
        headers = {"X-Request-ID": request_id}
        add_security_headers(headers)
        if exc.headers:
            headers.update(exc.headers)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "error": {
                    "code": error_code,
                    "message": exc.detail,
                    "details": {},
                    "request_id": request_id
                }
            },
            headers=headers
        )
