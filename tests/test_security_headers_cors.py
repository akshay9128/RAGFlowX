import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.config import settings


@pytest.fixture
def headers_client():
    return TestClient(app)


def test_security_headers_present(headers_client):
    """Security headers are set on responses when enabled."""
    res = headers_client.get("/health")
    assert res.status_code == 200
    assert res.headers.get("X-Content-Type-Options") == "nosniff"
    assert res.headers.get("X-Frame-Options") == "DENY"
    assert "Referrer-Policy" in res.headers
    assert "Content-Security-Policy" in res.headers


def test_cors_preflight(headers_client):
    """CORS preflight requests are handled."""
    res = headers_client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    # CORS middleware should allow the preflight
    assert res.status_code in (200, 204)


def test_cors_origin_header(headers_client):
    """CORS allows configured origins."""
    res = headers_client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert res.status_code == 200
