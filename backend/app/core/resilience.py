import time
import random
import logging
from typing import Callable, Any, Optional, Dict
from backend.app.config import settings

logger = logging.getLogger(__name__)


class CircuitBreakerOpenError(Exception):
    """Raised when a provider's circuit breaker is currently Open."""
    pass


class CircuitBreaker:
    """Circuit breaker per provider (Closed, Open, Half-Open)."""
    def __init__(self, provider_name: str, failure_threshold: int = 5, cooldown: float = 60.0, clock=time.time):
        self.provider_name = provider_name
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self.clock = clock

        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        self.failure_count = 0
        self.last_failure_time = 0.0

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = self.clock()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker for provider '{self.provider_name}' transitioned to OPEN after {self.failure_count} failures.")

    def check_state(self):
        if self.state == "OPEN":
            elapsed = self.clock() - self.last_failure_time
            if elapsed >= self.cooldown:
                self.state = "HALF-OPEN"
                logger.info(f"Circuit breaker for provider '{self.provider_name}' transitioned to HALF-OPEN.")
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker for '{self.provider_name}' is OPEN. Cooldown remaining: {int(self.cooldown - elapsed)}s.")


class ResilienceManager:
    """Manages circuit breakers and retry logic with exponential backoff and jitter."""
    def __init__(self, clock=time.time, sleeper=time.sleep):
        self.clock = clock
        self.sleeper = sleeper
        self.breakers: Dict[str, CircuitBreaker] = {}

    def get_breaker(self, provider_name: str) -> CircuitBreaker:
        if provider_name not in self.breakers:
            self.breakers[provider_name] = CircuitBreaker(
                provider_name=provider_name,
                failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                cooldown=settings.CIRCUIT_BREAKER_COOLDOWN,
                clock=self.clock
            )
        return self.breakers[provider_name]

    def execute_with_resilience(
        self,
        provider_name: str,
        func: Callable[..., Any],
        *args,
        **kwargs
    ) -> Any:
        breaker = self.get_breaker(provider_name)
        breaker.check_state()

        retries = settings.RESILIENCE_MAX_RETRIES
        base_delay = settings.RESILIENCE_BASE_DELAY
        max_delay = settings.RESILIENCE_MAX_DELAY
        use_jitter = settings.RESILIENCE_JITTER

        attempt = 0
        while True:
            attempt += 1
            is_transient = False
            last_exc: Optional[BaseException] = None
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except (TimeoutError, ConnectionError) as exc:
                last_exc = exc
                is_transient = True
            except Exception as exc:
                # Check for HTTP status codes in exception if available (e.g. 429, 5xx)
                status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                if status_code in (429, 500, 502, 503, 504):
                    last_exc = exc
                    is_transient = True
                else:
                    # Non-retryable error (auth, bad request, etc.)
                    breaker.record_failure()
                    raise exc

            if is_transient:
                breaker.record_failure()
                if attempt > retries:
                    logger.error(f"Provider '{provider_name}' failed after {retries} retries due to transient error.")
                    raise last_exc  # type: ignore[arg-type]

                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                if use_jitter:
                    delay = delay * (0.5 + random.random())

                logger.warning(f"Provider '{provider_name}' transient error on attempt {attempt}/{retries}. Retrying in {delay:.2f}s...")
                self.sleeper(delay)


resilience_manager = ResilienceManager()
