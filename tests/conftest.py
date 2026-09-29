import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


@pytest.fixture
def client():
    """Pytest fixture that returns a TestClient for FastAPI app."""
    return TestClient(app)
