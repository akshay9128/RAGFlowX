import os
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.security import api_key_store, hash_api_key


@pytest.fixture
def auth_client():
    # Enable auth for testing auth endpoints
    os.environ["AUTH_ENABLED"] = "true"
    os.environ["API_KEY_PEPPER"] = "test-pepper-12345678901234567890123456789012"
    # Re-instantiate app / settings or test with client
    from backend.app.config import settings
    settings.AUTH_ENABLED = True
    client = TestClient(app)
    yield client
    settings.AUTH_ENABLED = False
    os.environ["AUTH_ENABLED"] = "false"


def test_health_endpoint_stays_open_without_auth(auth_client):
    res = auth_client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_protected_endpoint_missing_key_returns_401(auth_client):
    res = auth_client.get("/api/v1/sessions")
    assert res.status_code == 401


def test_protected_endpoint_invalid_key_returns_401(auth_client):
    res = auth_client.get("/api/v1/sessions", headers={"X-API-Key": "invalid-key"})
    assert res.status_code == 401


def test_key_management_and_scopes(auth_client):
    # Clear any existing keys from the store (fresh state per test)
    api_key_store._keys.clear()

    # Create an admin key directly via api_key_store for testing
    raw_admin, record = api_key_store.create_key("Test Admin", ["admin", "read", "write"])

    # Test valid key access
    res = auth_client.get("/api/v1/sessions", headers={"X-API-Key": raw_admin})
    assert res.status_code == 200

    # Test listing keys (admin scope)
    res_keys = auth_client.get("/api/v1/keys", headers={"X-API-Key": raw_admin})
    assert res_keys.status_code == 200
    keys_data = res_keys.json()
    assert any(k["key_id"] == record.key_id for k in keys_data)

    # Create a read-only key
    raw_read, read_record = api_key_store.create_key("Test Reader", ["read"])

    # Read-only key should fail on admin endpoint (delete / revoke)
    res_revoke = auth_client.delete(f"/api/v1/keys/{record.key_id}", headers={"X-API-Key": raw_read})
    assert res_revoke.status_code == 403

    # Revoke key
    api_key_store.revoke_key(read_record.key_id)
    res_revoked_use = auth_client.get("/api/v1/sessions", headers={"X-API-Key": raw_read})
    assert res_revoked_use.status_code == 401
