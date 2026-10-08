# RAGFlowX 🚀
### Hybrid RAG System with Dense Search, BM25, Reranking, and Verified Citations

**RAGFlowX** is a production-oriented, real-time Retrieval-Augmented Generation (RAG) web application for processing, indexing, searching, and answering questions over user-uploaded PDF and text documents with verified citation sources.

---

## 💡 System Capabilities Built So Far

### 1. Core Architecture & FastAPI Foundation
- **Modular Micro-Services**: Built a clean, decoupled service architecture separated into ingestion, chunking, embeddings, retrieval, reranking, generation, and citations.
- **Environment Management**: Configuration driven by `pydantic-settings` with automatic directory creation for uploads and persistent data.
- **Health Monitoring**: Production health check endpoints (`/health`, `/api/v1/health`) for application liveness.

### 2. Document Ingestion & Page-Level Extraction
- **Multi-Format Uploads**: Support for PDF, TXT, and Markdown file uploads.
- **Validation Engine**: Strictly validates file extensions, size limits (up to 20MB), and empty uploads.
- **Page-by-Page Extraction**: Uses `pypdf` to extract text from PDF documents page-by-page while cleaning control characters and formatting artifacts.
- **Metadata Lineage**: Generates a unique `document_id` UUID for every upload and preserves document lineage (original `filename`, `content_type`, `file_size`, `total_pages`, `source` path, and `created_at` timestamp).

### 3. Text Chunking & Metadata Traceability
- **Recursive Boundary Splitting**: Intelligently splits page text into configurable chunk sizes (e.g. 500 characters) with overlap (e.g. 50 characters) respecting paragraphs (`\n\n`), line breaks (`\n`), sentences (`. `), and spaces (` `).
- **Metadata Preservation on Chunks**: Every chunk maintains traceability back to its source document: `chunk_id`, `document_id`, `filename`, `page_number`, `chunk_index`, and `source` file path.

### 4. Dense Embeddings & Vector Service
- **Modular Abstract Base Provider**: Abstract `BaseEmbeddingProvider` interface allowing seamless, runtime swapping between `SentenceTransformers`, `FastEmbed`, OpenAI, Gemini, or Mock providers without hardcoding.
- **Vector Generation**: Supports embedding generation for arbitrary query strings and full document chunk callsets into normalized 384-dimensional dense vectors.
- **Embedding Factory**: Dynamic factory instantiation based on `EMBEDDING_PROVIDER` and `EMBEDDING_MODEL_NAME` configuration.

### 5. Vector Database & Similarity Search
- **Modular Vector Store Interface**: Abstract `BaseVectorStore` interface supporting interchangeable vector database engines (`in_memory`, `chroma`, etc.).
- **Vectorized Similarity Search**: High-performance numpy-based Cosine Similarity calculation with score normalization `[0.0, 1.0]` and document scope filtering (`document_id`).
- **Persistent Vector Index**: Persistent JSON storage (`data/vector_store.json`) for indexed document vectors and chunk metadata.

### 6. Grounded LLM Generation & Basic RAG Pipeline
- **Modular LLM Generator**: Abstract `BaseLLMProvider` interface supporting Gemini API (`gemini-2.5-flash`), OpenAI, or deterministic `MockLLMProvider`.
- **Grounded Prompt Engine**: Strict prompt template enforcement ensuring answers are strictly derived from retrieved evidence chunks, with explicit fallbacks when documents lack required information.
- **Basic RAG Pipeline**: End-to-end question answering pipeline: User Question ➔ Dense Vector Retrieval ➔ Context Formatting ➔ LLM Generation ➔ Grounded Answer + Source Evidence.

### 7. Persistent Storage Engines
- **Document Store**: In-memory cache backed by persistent JSON files (`data/doc_<document_id>.json`) for document metadata and page text.
- **Chunk Store**: In-memory cache backed by persistent JSON files (`data/chunks_<document_id>.json`) for processed text chunks.
- **Vector Store**: In-memory numpy matrix index backed by persistent JSON file (`data/vector_store.json`) for dense embeddings and metadata.

### 8. Production API Endpoints
- `GET /health` & `GET /api/v1/health` — Liveness & status check.
- `POST /api/v1/documents/upload` — Upload and ingest PDF/text documents.
- `GET /api/v1/documents/` — List all ingested documents.
- `GET /api/v1/documents/{document_id}` — Retrieve metadata and extracted pages for a document.
- `DELETE /api/v1/documents/{document_id}` — Remove document and raw file from disk.
- `POST /api/v1/chunks/process/{document_id}` — Process document into configurable text chunks.
- `GET /api/v1/chunks/{document_id}` — Retrieve generated chunks and metadata for a document.
- `DELETE /api/v1/chunks/{document_id}` — Delete stored chunks for a document.
- `POST /api/v1/embeddings/text` — Generate dense vector embedding for single text string.
- `POST /api/v1/embeddings/chunks/{document_id}` — Generate dense vector embeddings for all document chunks.
- `GET /api/v1/embeddings/info` — Inspect active embedding model, provider, and dimensionality.
- `POST /api/v1/vector-store/index/{document_id}` — Generate embeddings and index document chunks into vector database.
- `POST /api/v1/vector-store/search` — Perform dense vector similarity search and return top_k relevant chunks with similarity scores.
- `GET /api/v1/vector-store/stats` — Inspect vector store statistics (total vectors, provider, dimension).
- `DELETE /api/v1/vector-store/document/{document_id}` — Delete all indexed vectors for a document.
- `POST /api/v1/rag/query` — Execute Basic RAG pipeline (dense retrieval + LLM generation).
- `GET /api/v1/rag/info` — Inspect active RAG configuration, LLM model, and providers.

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
│       │       ├── health.py    # Health check endpoint
│       │       ├── documents.py # Document ingestion endpoints
│       │       ├── chunks.py    # Text chunking endpoints
│       │       ├── embeddings.py# Dense embedding endpoints
│       │       ├── vector_store.py# Vector Database endpoints
│       │       └── rag.py       # Basic RAG endpoints
│       ├── services/            # Modular service components
│       │   ├── ingestion/       # Validation & PDF page text extraction
│       │   ├── chunking/        # Recursive text splitter & chunking service
│       │   ├── embeddings/      # Modular embedding providers (SentenceTransformers, Mock)
│       │   ├── retrieval/       # Dense vector search & RAG pipeline service
│       │   ├── reranking/       # Cross-encoder reranker
│       │   ├── generation/      # Modular LLM generation providers (Gemini, Mock)
│       │   └── citations/       # Citation verification engine
│       ├── models/              # Domain data models (document, chunk, embedding, search)
│       ├── schemas/             # Pydantic API request/response schemas (rag, search, chunk, document)
│       ├── database/            # DocumentStore, ChunkStore & VectorStore persistence
│       └── utils/               # Helper functions & logging setup
├── data/                        # Persistent JSON data storage & vector_store.json
├── uploads/                     # Raw file upload storage
├── tests/                       # Automated pytest test suite
│   ├── conftest.py
│   ├── test_health.py
│   ├── test_documents.py
│   ├── test_chunks.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   └── test_rag.py
├── run.py                       # Lightweight server entrypoint script (python run.py)
├── pytest.ini                   # Pytest configuration
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
python run.py
```

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 🧪 Testing

Run pytest to execute all automated unit & integration tests:

```bash
pytest
```
