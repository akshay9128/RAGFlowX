# RAGFlowX Testing Documentation

Welcome to the RAGFlowX testing guide. This document outlines the test architecture, test tiers, fixtures, isolation practices, security regression testing, and CI pipeline execution.

## Test Tiers & Markers

RAGFlowX uses strict pytest markers (`pytest.ini` with `strict-markers = true`) to categorize tests into distinct execution tiers:

1. **`default`**: Fast, offline, deterministic mock provider tests (unit tests, embedding mock tests, vector store operations).
2. **`e2e`**: Full application end-to-end integration tests (FastAPI TestClient, store isolation, multi-turn chat sessions, security regression matrix, fixtures and manifest extraction).
3. **`quality`**: Local model embedding and reranker evaluation tests.
4. **`live`**: Real Gemini model tests (skipped by default unless explicitly enabled).

---

## Running Tests

### Running All Tiers via Script
You can execute the automated cross-platform test runner script:
```bash
python scripts/run_all.py
```

### Running Specific Tiers via Pytest
- Run default unit tests:
  ```bash
  pytest -m default
  ```
- Run end-to-end integration tests:
  ```bash
  pytest -m e2e
  ```
- Run specific test file:
  ```bash
  pytest tests/e2e/test_pipeline_and_multiturn.py -v
  ```

---

## Test Fixtures & Manifest (`tests/fixtures/manifest.json`)

Test fixtures include multi-page PDFs, text documents, Markdown files, unanswerable questions, and prompt-injection defense payloads. The fixture manifest (`tests/fixtures/manifest.json`) defines expected document structures, pages, facts, and security edge cases (empty files, corrupt PDFs, wrong magic bytes).

---

## Security Regression & Isolation

- **Isolation**: E2E tests utilize temporary directories (`tmp_path`) for document, chunk, session, and API key stores to ensure zero pollution between test runs.
- **Authentication**: Dynamic per-request evaluation of `AUTH_ENABLED` and salted SHA-256 API key hashing with runtime peppers.
- **Rate Limiting**: Enforces request rate limits and returns `Retry-After` headers on 429 responses.
- **Request Tracing**: Injects `X-Request-ID` headers across all API responses for end-to-end observability.

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) automatically runs the test suite across Python 3.10, 3.11, and 3.12 on every push and pull request.
