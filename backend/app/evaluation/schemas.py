from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class BenchmarkItem(BaseModel):
    id: str
    question: str
    gold_answer: str
    relevant_chunks: List[str] = Field(default_factory=list, description="List of ground truth chunk IDs or empty if relying on doc/page/span")
    document_filename: Optional[str] = None
    page_number: Optional[int] = None
    relevant_spans: List[str] = Field(default_factory=list)
    answerable: bool = True
    split: str = "dev"  # dev or test
    tags: List[str] = Field(default_factory=list)

class BenchmarkDataset(BaseModel):
    name: str
    version: str = "1.0"
    description: str = ""
    documents: List[Dict[str, Any]] = Field(default_factory=list, description="Sample documents to ingest for evaluation")
    items: List[BenchmarkItem] = Field(default_factory=list)

class AblationConfig(BaseModel):
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    embedding_provider: Optional[str] = None
    embedding_model_name: Optional[str] = None
    rrf_k: Optional[int] = None
    top_k: Optional[int] = None
    candidate_pool_size: Optional[int] = None
    reranker_enabled: Optional[bool] = None
    reranker_model: Optional[str] = None
    retrieval_mode: str = "hybrid_rerank"  # dense, bm25, hybrid, hybrid_rerank

class RetrievalMetricsResult(BaseModel):
    precision_at_k: Dict[int, float] = Field(default_factory=dict)
    recall_at_k: Dict[int, float] = Field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: Dict[int, float] = Field(default_factory=dict)
    hit_rate_at_k: Dict[int, float] = Field(default_factory=dict)
    candidate_pool_recall: float = 0.0

class GenerationMetricsResult(BaseModel):
    groundedness_score: float = 0.0
    answer_relevance: float = 0.0
    citation_accuracy: float = 0.0
    token_f1: float = 0.0
    exact_match: bool = False
    correct_abstention: Optional[bool] = None
    confidence_score: float = 0.0

class LatencyMetricsResult(BaseModel):
    stages: Dict[str, Dict[str, float]] = Field(default_factory=dict)  # stage -> {mean, p50, p95, p99, min, max, count}
    total_duration_ms: float = 0.0

class QuestionEvaluationResult(BaseModel):
    item_id: str
    question: str
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    generated_answer: str = ""
    gold_answer: str = ""
    answerable: bool = True
    retrieval_metrics: RetrievalMetricsResult = Field(default_factory=RetrievalMetricsResult)
    generation_metrics: GenerationMetricsResult = Field(default_factory=GenerationMetricsResult)
    latency: LatencyMetricsResult = Field(default_factory=LatencyMetricsResult)
    failure_cause: Optional[str] = None  # retrieval_miss, rerank_loss, generation_error, none

class RunReport(BaseModel):
    run_id: str
    timestamp: str
    git_commit: Optional[str] = None
    dataset_name: str
    split: str
    config: AblationConfig
    aggregate_metrics: Dict[str, Any] = Field(default_factory=dict)
    failure_attribution_summary: Dict[str, int] = Field(default_factory=dict)
    latency_summary: Dict[str, Any] = Field(default_factory=dict)
    question_results: List[QuestionEvaluationResult] = Field(default_factory=list)

class RunComparisonResult(BaseModel):
    run_id_a: str
    run_id_b: str
    deltas: Dict[str, float] = Field(default_factory=dict)
    regressions_detected: List[str] = Field(default_factory=list)
    within_noise_band: List[str] = Field(default_factory=list)
    passed: bool = True
