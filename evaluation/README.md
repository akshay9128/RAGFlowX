# RAGFlowX Evaluation Suite

The RAGFlowX Evaluation Suite provides objective quality measurement across retrieval, generation, and latency metrics.

## Features
- **Retrieval Metrics:** Precision@k, Recall@k, MRR, nDCG@k, Hit Rate@k, and candidate pool recall.
- **Generation Metrics:** Groundedness, answer relevance, citation accuracy, token F1, exact match, and abstention checks.
- **Latency Tracking:** Per-stage and end-to-end latency percentiles (p50, p95, p99).
- **Failure Attribution:** Automatically classifies retrieval misses, rerank losses, and generation errors.
- **Ablation & Comparison:** Test parameter sweeps (chunk size, overlap, retrieval modes, reranker toggle) in isolated indexes and compare runs against baselines.

## How to Run
Run tests via pytest:
```bash
pytest tests/test_evaluation_metrics.py tests/test_evaluation_runner.py
```
