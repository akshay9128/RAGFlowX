import uuid
import time
import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

async def request_tracing_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start_time = time.time()

    try:
        response = await call_next(request)
        duration = time.time() - start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration:.4f}"
        return response
    except Exception as exc:
        duration = time.time() - start_time
        logger.exception(f"Unhandled exception during request {request_id}: {exc}")
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
            headers={"X-Request-ID": request_id}
        )

def register_exception_handlers(app):
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
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
            headers={"X-Request-ID": request_id}
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        code_map = {
            400: "BAD_REQUEST",
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            422: "UNPROCESSABLE_ENTITY",
        }
        error_code = code_map.get(exc.status_code, "HTTP_ERROR")
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
            headers={"X-Request-ID": request_id}
        )
