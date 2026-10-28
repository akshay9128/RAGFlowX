import os
from pathlib import Path
import pytest
from pydantic import ValidationError
from backend.app.config import Settings


def test_config_auth_disabled_no_pepper_succeeds(monkeypatch):
    # Covers: app starting with AUTH_ENABLED=false and no pepper using monkeypatch and _env_file=None
    monkeypatch.delenv("API_KEY_PEPPER", raising=False)
    monkeypatch.delenv("BOOTSTRAP_API_KEY", raising=False)
    cfg = Settings(_env_file=None, AUTH_ENABLED=False, API_KEY_PEPPER=None)
    assert cfg.AUTH_ENABLED is False
    assert cfg.API_KEY_PEPPER is None


def test_config_auth_pepper_missing_raises_error():
    # Covers: missing pepper when AUTH_ENABLED=True
    with pytest.raises(ValueError, match="API_KEY_PEPPER"):
        Settings(_env_file=None, AUTH_ENABLED=True, API_KEY_PEPPER="")


def test_config_auth_pepper_too_short_raises_error():
    # Covers: short pepper (< 32 chars)
    with pytest.raises(ValueError, match="API_KEY_PEPPER"):
        Settings(_env_file=None, AUTH_ENABLED=True, API_KEY_PEPPER="short-pepper-not-32-chars-long")


@pytest.mark.parametrize("placeholder", [
    "your-some-very-long-custom-string-that-has-over-thirty-two-chars",
    "placeholder-some-very-long-custom-string-over-thirty-two-chars",
    "changeme-some-very-long-custom-string-over-thirty-two-chars",
    "example-some-very-long-custom-string-over-thirty-two-chars",
    "min-32-some-very-long-custom-string-over-thirty-two-chars",
    "min-16-some-very-long-custom-string-over-thirty-two-chars",
])
def test_config_auth_pepper_placeholder_patterns_raise_error(placeholder):
    # Covers: each placeholder pattern with >=32 chars and >=10 distinct chars apart from pattern
    with pytest.raises(ValueError, match="API_KEY_PEPPER"):
        Settings(_env_file=None, AUTH_ENABLED=True, API_KEY_PEPPER=placeholder)


def test_config_auth_pepper_low_entropy_raises_error():
    # Covers: low-entropy values (< 10 distinct characters)
    with pytest.raises(ValueError, match="API_KEY_PEPPER"):
        Settings(_env_file=None, AUTH_ENABLED=True, API_KEY_PEPPER="a" * 32)


def test_config_auth_pepper_valid_succeeds():
    # Covers: valid pepper
    valid_pepper = "abcdefghijklmnopqrstuvwxyz123456"  # 32 chars, 32 distinct chars
    cfg = Settings(_env_file=None, AUTH_ENABLED=True, API_KEY_PEPPER=valid_pepper)
    assert cfg.AUTH_ENABLED is True
    assert cfg.API_KEY_PEPPER == valid_pepper


def test_config_bootstrap_key_too_short_raises_error():
    # Covers: short bootstrap key (< 16 chars)
    with pytest.raises(ValueError, match="BOOTSTRAP_API_KEY"):
        Settings(_env_file=None, AUTH_ENABLED=True, API_KEY_PEPPER="abcdefghijklmnopqrstuvwxyz123456", BOOTSTRAP_API_KEY="short")


@pytest.mark.parametrize("bootstrap_placeholder", [
    "your-bootstrap-key-placeholder-val",
    "placeholder-bootstrap-key-val",
    "changeme-bootstrap-key-val",
    "example-bootstrap-key-val",
    "min-16-bootstrap-key-val",
])
def test_config_bootstrap_key_placeholder_patterns_raise_error(bootstrap_placeholder):
    # Covers: bootstrap key placeholder patterns
    with pytest.raises(ValueError, match="BOOTSTRAP_API_KEY"):
        Settings(_env_file=None, AUTH_ENABLED=True, API_KEY_PEPPER="abcdefghijklmnopqrstuvwxyz123456", BOOTSTRAP_API_KEY=bootstrap_placeholder)


def test_config_bootstrap_key_valid_succeeds():
    # Covers: valid bootstrap key
    valid_pepper = "abcdefghijklmnopqrstuvwxyz123456"
    valid_bootstrap = "valid-bootstrap-key-16"
    cfg = Settings(_env_file=None, AUTH_ENABLED=True, API_KEY_PEPPER=valid_pepper, BOOTSTRAP_API_KEY=valid_bootstrap)
    assert cfg.BOOTSTRAP_API_KEY == valid_bootstrap


def test_config_error_messages_do_not_echo_secret_value():
    # Covers: error messages not echoing the raw secret value
    secret_val = "your-very-secret-pepper-that-fails-validation"
    with pytest.raises(ValueError) as exc_info:
        Settings(_env_file=None, AUTH_ENABLED=True, API_KEY_PEPPER=secret_val)
    error_msg = str(exc_info.value)
    assert secret_val not in error_msg
    assert "API_KEY_PEPPER" in error_msg


def test_config_env_example_values_fail_validation():
    # Covers: reading real .env.example file and asserting its pepper and bootstrap values fail validation
    env_example_path = Path(".env.example")
    assert env_example_path.exists()

    env_vars = {}
    for line in env_example_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            env_vars[key.strip()] = val.strip().strip('"').strip("'")

    pepper = env_vars.get("API_KEY_PEPPER")
    bootstrap = env_vars.get("BOOTSTRAP_API_KEY")
    auth_enabled = env_vars.get("AUTH_ENABLED", "True").lower() == "true"

    assert pepper is not None
    assert bootstrap is not None

    with pytest.raises(ValueError):
        Settings(
            _env_file=None,
            AUTH_ENABLED=auth_enabled,
            API_KEY_PEPPER=pepper,
            BOOTSTRAP_API_KEY=bootstrap
        )
