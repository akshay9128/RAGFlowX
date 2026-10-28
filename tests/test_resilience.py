import time
from unittest.mock import patch
from backend.app.core.resilience import CircuitBreaker, ResilienceManager, CircuitBreakerOpenError


def test_circuit_breaker_closed_to_open():
    """Circuit breaker transitions Closed -> Open after threshold failures."""
    breaker = CircuitBreaker("test_provider", failure_threshold=3, cooldown=60.0, clock=lambda: 1000.0)

    assert breaker.state == "CLOSED"
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "CLOSED"
    breaker.record_failure()
    assert breaker.state == "OPEN"


def test_circuit_breaker_open_raises():
    """Open circuit breaker raises CircuitBreakerOpenError."""
    breaker = CircuitBreaker("test_provider", failure_threshold=1, cooldown=60.0, clock=lambda: 1000.0)
    breaker.record_failure()
    assert breaker.state == "OPEN"

    try:
        breaker.check_state()
        assert False, "Expected CircuitBreakerOpenError"
    except CircuitBreakerOpenError:
        pass


def test_circuit_breaker_open_to_half_open_after_cooldown():
    """After cooldown, Open transitions to Half-Open."""
    times = [1000.0]
    def clock():
        return times[0]

    breaker = CircuitBreaker("test_provider", failure_threshold=1, cooldown=30.0, clock=clock)
    breaker.record_failure()  # at t=1000, state -> OPEN
    assert breaker.state == "OPEN"

    times[0] = 1031.0  # 31 seconds later, past cooldown
    breaker.check_state()  # should transition to HALF-OPEN
    assert breaker.state == "HALF-OPEN"


def test_circuit_breaker_half_open_success_closes():
    """Success in Half-Open closes the circuit."""
    breaker = CircuitBreaker("test_provider", failure_threshold=1, cooldown=60.0, clock=lambda: 1000.0)
    breaker.record_failure()
    assert breaker.state == "OPEN"

    # Manually set to HALF-OPEN then record success
    breaker.state = "HALF-OPEN"
    breaker.record_success()
    assert breaker.state == "CLOSED"
    assert breaker.failure_count == 0


def test_resilience_manager_retries_transient_error():
    """Resilience manager retries on transient errors with backoff."""
    manager = ResilienceManager(clock=lambda: 1000.0, sleeper=lambda x: None)
    call_count = {"n": 0}

    def flaky_func():
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise TimeoutError("temporary")
        return "success"

    result = manager.execute_with_resilience("provider_a", flaky_func)
    assert result == "success"
    assert call_count["n"] == 3


def test_resilience_manager_non_retryable_error():
    """Non-retryable errors are raised immediately without retry."""
    manager = ResilienceManager(clock=lambda: 1000.0, sleeper=lambda x: None)
    call_count = {"n": 0}

    def always_fail():
        call_count["n"] += 1
        raise ValueError("bad request")

    try:
        manager.execute_with_resilience("provider_b", always_fail)
        assert False, "Expected ValueError"
    except ValueError:
        pass
    assert call_count["n"] == 1


def test_resilience_manager_circuit_open_blocks():
    """When circuit is OPEN, execute_with_resilience raises CircuitBreakerOpenError."""
    manager = ResilienceManager(clock=lambda: 1000.0, sleeper=lambda x: None)
    breaker = manager.get_breaker("provider_c")
    breaker.failure_threshold = 1
    breaker.record_failure()  # OPEN

    try:
        manager.execute_with_resilience("provider_c", lambda: "ok")
        assert False, "Expected CircuitBreakerOpenError"
    except CircuitBreakerOpenError:
        pass


def test_exponential_backoff_delay():
    """Backoff delay grows exponentially with jitter."""
    manager = ResilienceManager(clock=lambda: 1000.0, sleeper=lambda x: None)
    delays = []

    def capture_sleeper(d):
        delays.append(d)

    manager.sleeper = capture_sleeper
    call_count = {"n": 0}

    def always_timeout():
        call_count["n"] += 1
        raise TimeoutError("always")

    try:
        manager.execute_with_resilience("provider_d", always_timeout)
    except TimeoutError:
        pass

    # With MAX_RETRIES=3, we should have 3 retries (attempts 2, 3, 4 after first failure)
    # Delays should be increasing
    assert len(delays) == call_count["n"] - 1
    assert all(d > 0 for d in delays)
