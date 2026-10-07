# Hybrid RAG System 

Production-oriented real-time web application for Retrieval-Augmented Generation (RAG) over user-uploaded PDF/documents featuring hybrid retrieval, reranking, LLM generation, and verified citation sources.

## 🚀 Features & Architecture

- **Document Ingestion & Parsing**: PDF/TXT/MD upload, validation, page-level text extraction with `pypdf`, metadata preservation (`document_id`, `filename`, `page_number`, `source`).
- **Document Store**: Persistent storage for processed document metadata and page text.
- **Text Chunking**: Configurable semantic recursive chunking (`chunk_size`, `chunk_overlap`) with paragraph/sentence boundary splitting and complete metadata preservation (`chunk_id`, `document_id`, `filename`, `page_number`, `chunk_index`, `source`).
- **Chunk Store**: Persistent storage for document text chunks in `data/chunks_<document_id>.json`.
- **Dense Embeddings & Vector Search**: Modular embedding services & vector storage.
- **BM25 Lexical Search**: Keyword retrieval for exact terms, names, and rare tokens.
- **Hybrid Search (RRF)**: Reciprocal Rank Fusion combining dense semantic & BM25 lexical results.
- **Reranking**: Post-retrieval cross-encoder re-ranking for optimal context precision.
- **Grounded LLM Generation**: Strict prompt-grounded context generation with fallback guarantees.
- **Verified Source Citations**: Document name, page number, and chunk traceability.
- **FastAPI Web API**: Production REST endpoints for document management, search, and chat.

---

## 📁 Folder Structure

```text
.
├── backend/
│   └── app/
│       ├── main.py              # FastAPI application entrypoint
│       ├── config.py            # Environment & app configuration
│       ├── api/                 # API routers & endpoint definitions
│       │   ├── router.py
│       │   └── routes/
│       │       └── health.py
│       ├── services/            # Modular service components
│       │   ├── ingestion/       # PDF parsing & text extraction
│       │   ├── chunking/        # Text splitting & chunking
│       │   ├── embeddings/      # Dense embedding service
│       │   ├── retrieval/       # Dense, BM25 & Hybrid search
│       │   ├── reranking/       # Cross-encoder reranker
│       │   ├── generation/      # Grounded LLM generation
│       │   └── citations/       # Citation verification engine
│       ├── models/              # Core domain data models
│       ├── schemas/             # Pydantic API request/response schemas
│       ├── database/            # Vector & metadata persistence
│       └── utils/               # Helper functions & logging setup
├── tests/                       # Unit & integration test suite
├── .env.example                 # Environment variable template
├── .gitignore                   # Git ignore settings
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🛠️ Setup & Execution

### 1. Environment Setup

Ensure Python 3.10+ is installed. Activate your virtual environment and install dependencies:

```bash
# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install required packages
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

### 3. Run the Backend API Server

```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 🧪 Testing

Run pytest to execute basic unit & health check tests:

```bash
python -m pytest
```
