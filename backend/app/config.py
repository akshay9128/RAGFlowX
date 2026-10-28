from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings using pydantic-settings."""
    APP_NAME: str = "RAGFlowX"
    APP_ENV: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_PREFIX: str = "/api/v1"

    # Project directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    DATA_DIR: Path = BASE_DIR / "data"

    # Embedding Settings
    EMBEDDING_PROVIDER: str = "sentence-transformers"  # sentence-transformers or mock
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Vector Database Settings
    VECTOR_STORE_PROVIDER: str = "in_memory"  # in_memory or chroma
    VECTOR_STORE_DIR: Path = DATA_DIR / "vector_store"

    # LLM & RAG Settings
    LLM_PROVIDER: str = "mock"  # mock, gemini, or openai
    LLM_MODEL_NAME: str = "gemini-2.5-flash"
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    RAG_TOP_K: int = 5

    # Phase 10 — Advanced Grounded LLM Generation feature flags
    RAG_ENABLE_CHAIN_OF_THOUGHT: bool = True
    RAG_ENABLE_CITATIONS: bool = True
    RAG_ENABLE_CONFIDENCE_SCORING: bool = True

    # Multi-provider failover (Gemini -> OpenAI -> Mock)
    LLM_FAILOVER_ENABLED: bool = True
    LLM_FAILOVER_PROVIDERS: str = "gemini,mock"

    # Phase 16 — Security & Reliability Settings
    AUTH_ENABLED: bool = True
    BOOTSTRAP_API_KEY: Optional[str] = None
    API_KEY_PEPPER: Optional[str] = None  # Required when AUTH_ENABLED=True; no default

    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_DEFAULT_REQUESTS: int = 60
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    RATE_LIMIT_EXPENSIVE_REQUESTS: int = 10
    RATE_LIMIT_AUTH_FAIL_LIMIT: int = 5

    TRUSTED_PROXY_ENABLED: bool = False
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000"
    SECURITY_HEADERS_ENABLED: bool = True
    LOG_REDACTION_ENABLED: bool = True

    UPLOAD_MAX_PAGE_COUNT: int = 500
    UPLOAD_MAX_EXTRACTED_CHARS: int = 2000000
    UPLOAD_MAX_BODY_SIZE_BYTES: int = 25 * 1024 * 1024  # 25MB
    EXTRACTION_TIMEOUT_SECONDS: int = 30
    ANTIVIRUS_ENABLED: bool = False
    PDF_RISKY_FEATURE_MODE: str = "reject"  # reject or flag

    PROMPT_INJECTION_MODE: str = "neutralize"  # off, flag, neutralize, block

    RESILIENCE_MAX_RETRIES: int = 3
    RESILIENCE_BASE_DELAY: float = 1.0
    RESILIENCE_MAX_DELAY: float = 60.0
    RESILIENCE_JITTER: bool = True
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_COOLDOWN: float = 60.0
    LLM_CONNECT_TIMEOUT: float = 10.0
    LLM_READ_TIMEOUT: float = 60.0

    MAX_QUERY_LENGTH: int = 1000
    MAX_TOP_K: int = 20
    MAX_SESSION_MESSAGES: int = 100

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def model_post_init(self, __context):
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

        forbidden_snippets = ["placeholder", "changeme", "example", "min-32", "min-16"]
        allowed_prompt_injection = {"off", "flag", "neutralize", "block"}
        if self.PROMPT_INJECTION_MODE not in allowed_prompt_injection:
            raise ValueError(f"Security Error: PROMPT_INJECTION_MODE must be one of {allowed_prompt_injection}")

        allowed_pdf_risky = {"reject", "flag"}
        if self.PDF_RISKY_FEATURE_MODE not in allowed_pdf_risky:
            raise ValueError(f"Security Error: PDF_RISKY_FEATURE_MODE must be one of {allowed_pdf_risky}")

        positive_ints = {
            "UPLOAD_MAX_PAGE_COUNT": self.UPLOAD_MAX_PAGE_COUNT,
            "UPLOAD_MAX_EXTRACTED_CHARS": self.UPLOAD_MAX_EXTRACTED_CHARS,
            "UPLOAD_MAX_BODY_SIZE_BYTES": self.UPLOAD_MAX_BODY_SIZE_BYTES,
            "EXTRACTION_TIMEOUT_SECONDS": self.EXTRACTION_TIMEOUT_SECONDS,
            "MAX_QUERY_LENGTH": self.MAX_QUERY_LENGTH,
            "MAX_TOP_K": self.MAX_TOP_K,
            "MAX_SESSION_MESSAGES": self.MAX_SESSION_MESSAGES,
            "RESILIENCE_BASE_DELAY": self.RESILIENCE_BASE_DELAY,
            "RESILIENCE_MAX_DELAY": self.RESILIENCE_MAX_DELAY,
            "CIRCUIT_BREAKER_FAILURE_THRESHOLD": self.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            "CIRCUIT_BREAKER_COOLDOWN": self.CIRCUIT_BREAKER_COOLDOWN,
            "LLM_CONNECT_TIMEOUT": self.LLM_CONNECT_TIMEOUT,
            "LLM_READ_TIMEOUT": self.LLM_READ_TIMEOUT,
        }
        for name, val in positive_ints.items():
            if val is None or val <= 0:
                raise ValueError(f"Configuration Error: {name} must be a positive number (> 0).")

        if self.RESILIENCE_MAX_RETRIES is None or self.RESILIENCE_MAX_RETRIES < 0:
            raise ValueError("Configuration Error: RESILIENCE_MAX_RETRIES must be >= 0.")

        if self.AUTH_ENABLED:
            pepper_val = str(self.API_KEY_PEPPER or "").strip()
            if (
                not pepper_val
                or pepper_val.lower().startswith("your-")
                or any(sub in pepper_val.lower() for sub in forbidden_snippets)
                or len(set(pepper_val)) < 10
                or len(pepper_val) < 32
            ):
                raise ValueError(
                    "Security Error: AUTH_ENABLED is True, but API_KEY_PEPPER is missing, "
                    "uses a placeholder or example value, has fewer than 10 distinct characters, or is shorter than 32 characters."
                )

        if self.BOOTSTRAP_API_KEY:
            bootstrap_val = str(self.BOOTSTRAP_API_KEY).strip()
            if (
                bootstrap_val.lower().startswith("your-")
                or any(sub in bootstrap_val.lower() for sub in forbidden_snippets)
                or len(set(bootstrap_val)) < 10
                or len(bootstrap_val) < 16
            ):
                raise ValueError(
                    "Security Error: BOOTSTRAP_API_KEY uses a placeholder or example value, "
                    "has fewer than 10 distinct characters, or is shorter than 16 characters."
                )


settings = Settings()
