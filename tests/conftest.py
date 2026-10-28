import os
import secrets
os.environ["AUTH_ENABLED"] = "false"
os.environ["RATE_LIMIT_ENABLED"] = "false"
os.environ["API_KEY_PEPPER"] = secrets.token_hex(32)

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


@pytest.fixture
def client():
    """Pytest fixture that returns a TestClient for FastAPI app."""
    return TestClient(app)
