import time
import logging
from typing import Dict, Tuple, Optional
from fastapi import Request, Response, HTTPException, status
from backend.app.config import settings

logger = logging.getLogger(__name__)


class RateLimiterStore:
    """Swappable storage interface for rate limiting (in-memory now, Redis-ready later)."""
    def is_allowed(self, identifier: str, max_requests: int, window_seconds: int, current_time: float) -> Tuple[bool, int, int, float]:
        raise NotImplementedError


class InMemoryRateLimiterStore(RateLimiterStore):
    def __init__(self):
        # identifier -> list of timestamps
        self.windows: Dict[str, list] = {}
        # identifier -> failure record for auth brute-forcing (timestamps)
        self.auth_failures: Dict[str, list] = {}

    def is_allowed(self, identifier: str, max_requests: int, window_seconds: int, current_time: float) -> Tuple[bool, int, int, float]:
        if identifier not in self.windows:
            self.windows[identifier] = []

        timestamps = self.windows[identifier]
        # Filter out timestamps older than window
        cutoff = current_time - window_seconds
        timestamps = [t for t in timestamps if t > cutoff]
        self.windows[identifier] = timestamps

        remaining = max(0, max_requests - len(timestamps))
        reset_time = current_time + window_seconds if not timestamps else timestamps[0] + window_seconds

        if len(timestamps) >= max_requests:
            return False, max_requests, 0, reset_time

        timestamps.append(current_time)
        return True, max_requests, remaining - 1, reset_time

    def record_auth_failure(self, ip: str, current_time: float, window_seconds: int = 300) -> int:
        if ip not in self.auth_failures:
            self.auth_failures[ip] = []
        cutoff = current_time - window_seconds
        fails = [t for t in self.auth_failures[ip] if t > cutoff]
        fails.append(current_time)
        self.auth_failures[ip] = fails
        return len(fails)

    def is_auth_throttled(self, ip: str, current_time: float, window_seconds: int = 300, max_fails: int = 5) -> bool:
        if ip not in self.auth_failures:
            return False
        cutoff = current_time - window_seconds
        fails = [t for t in self.auth_failures[ip] if t > cutoff]
        self.auth_failures[ip] = fails
        return len(fails) >= max_fails


rate_limiter_store = InMemoryRateLimiterStore()


def get_client_ip(request: Request) -> str:
    if settings.TRUSTED_PROXY_ENABLED:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # First IP in proxy chain
            return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


async def check_rate_limit(request: Request, response: Response, clock=time.time) -> None:
    if not settings.RATE_LIMIT_ENABLED or not settings.AUTH_ENABLED:
        return

    # Bypass public/health paths
    path = request.url.path
    if path == "/" or any(path.startswith(p) for p in ["/health", "/docs", "/redoc", "/openapi.json", "/static", "/ui"]):
        return

    current_time = clock()
    client_ip = get_client_ip(request)

    # Check auth failure throttling per IP
    if rate_limiter_store.is_auth_throttled(client_ip, current_time, window_seconds=300, max_fails=settings.RATE_LIMIT_AUTH_FAIL_LIMIT):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed authentication attempts from this IP. Temporarily throttled.",
            headers={"Retry-After": "300"}
        )

    # Determine identifier and limit
    # If API key authenticated, use key_id; otherwise use client IP
    api_key_record = getattr(request.state, "api_key_record", None)
    identifier = f"key_{api_key_record.key_id}" if api_key_record and api_key_record.key_id not in ("dev-bypass", "public") else f"ip_{client_ip}"

    # Determine if route is expensive (upload, chat/query, process)
    expensive_paths = ["/upload", "/process", "/chat", "/rag/query", "/rag/query/verified", "/rag/verify"]
    is_expensive = any(ep in path for ep in expensive_paths)

    max_reqs = settings.RATE_LIMIT_EXPENSIVE_REQUESTS if is_expensive else settings.RATE_LIMIT_DEFAULT_REQUESTS
    window = settings.RATE_LIMIT_WINDOW_SECONDS

    allowed, limit, remaining, reset = rate_limiter_store.is_allowed(identifier, max_reqs, window, current_time)

    # Set standard headers
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(int(reset))

    if not allowed:
        retry_after = int(reset - current_time)
        retry_after_str = str(max(1, retry_after))
        response.headers["Retry-After"] = retry_after_str
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": retry_after_str}
        )
