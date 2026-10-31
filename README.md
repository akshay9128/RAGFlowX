<div align="center">

# 🚀 RAGFlowX

### Production-Grade Hybrid RAG & Advanced Retrieval System
*Built with FastAPI, Dense + BM25 Hybrid Search, Cross-Encoder Reranking, Prompt Shielding, and Verified Citations*

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Production%20Ready-success.svg)]()

</div>

---

## 🌟 Overview

**RAGFlowX** is a modular, high-performance Retrieval-Augmented Generation (RAG) backend and API service engineered for enterprise-grade documentation intelligence. By combining dense semantic vector embeddings with sparse BM25 keyword matching, reciprocal rank fusion (RRF), cross-encoder reranking, and rigorous prompt shielding, RAGFlowX delivers accurate, fully grounded answers backed by verified claim-level source citations.

---

## ✨ Core Features

| Feature Category | Capabilities & Mechanisms |
| :--- | :--- |
| **📄 Ingestion & Chunking** | Multi-format support (PDF, TXT, MD), page-level extraction via `pypdf`, recursive text splitting preserving sentence boundaries and lineage. |
| **🔍 Hybrid Retrieval** | Dual-path retrieval combining SentenceTransformers dense vector search (cosine similarity) with `rank-bm25` keyword search and RRF fusion. |
| **⚡ Reranking & RAG** | Cross-encoder passage reranking, Chain-of-Thought reasoning (`use_chain_of_thought`), and dynamic confidence scoring. |
| **🛡️ Enterprise Security** | Hashed API key authentication (salted SHA-256 + PEPPER), sliding-window rate limiting (`Retry-After`), prompt injection defense (Prompt Shield), and security headers. |
| **🔄 Resilience & Tracing** | Circuit breakers, exponential backoff with jitter, log redaction, and `X-Request-ID` tracing across requests. |

---

## 🛠️ Tech Stack

- **Framework**: FastAPI (Asynchronous Python API)
- **Validation & Settings**: Pydantic / Pydantic Settings
- **Embeddings & Reranking**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Sparse Search**: `rank-bm25`
- **Vector Math**: NumPy Cosine Similarity & Indexing
- **Storage**: Persistent JSON-backed stores (`data/`, `uploads/`)

---

## 🏗️ System Architecture

```mermaid
graph TD
    User([User / Client]) --> API[FastAPI Endpoints]
    API --> Auth[Security & Rate Limiter]
    Auth --> Pipeline[RAG / Hybrid Service]
    
    subgraph Ingestion & Chunking
        Pipeline --> Ingest[Document Ingestion & pypdf]
        Ingest --> Chunk[Recursive Text Splitter]
    end

    subgraph Indexing & Retrieval
        Chunk --> Embed[Embedding Provider]
        Chunk --> BM25[BM25 Indexer]
        Embed --> VectorStore[(Vector Store)]
        BM25 --> SparseStore[(BM25 Store)]
        
        Pipeline --> DenseSearch[Dense Vector Search]
        Pipeline --> SparseSearch[BM25 Search]
        DenseSearch & SparseSearch --> Hybrid[Hybrid Fusion]
    end

    subgraph Generation & Verification
        Hybrid --> Rerank[Cross-Encoder Reranker]
        Rerank --> LLM[LLM Generation Provider / Failover]
        LLM --> Verify[Citation Verification Engine]
        Verify --> Response[Grounded Response + Citations]
    end

    Response --> User
```

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.10+
- Pip package manager

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/akshay9128/RAGFlowX.git
cd RAGFlowX

python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
# source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
```

### 4. Run the Server
```bash
python run.py
```
- **API Root**: `http://localhost:8000`
- **Swagger Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 📋 Key API Endpoints

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Health check & status | No |
| `POST` | `/api/v1/documents/upload` | Upload & ingest document (PDF/TXT/MD) | Yes |
| `GET` | `/api/v1/documents/` | List all ingested documents | Yes |
| `POST` | `/api/v1/chunks/process/{id}` | Process document into text chunks | Yes |
| `POST` | `/api/v1/vector-store/index/{id}` | Index chunks into vector store | Yes |
| `POST` | `/api/v1/hybrid/search` | Execute hybrid search (dense + sparse) | Yes |
| `POST` | `/api/v1/rerank/cross-encoder` | Rerank retrieved passages | Yes |
| `POST` | `/api/v1/rag/query` | Execute grounded RAG pipeline query | Yes |
| `GET` | `/api/v1/rag/info` | Inspect active RAG config & models | Yes |

---

## 🧪 Testing

Run the comprehensive test suite with `pytest`:
```bash
python -m pytest
```
All test cases across unit, e2e, and quality tiers execute and pass successfully.

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is proprietary and confidential.
