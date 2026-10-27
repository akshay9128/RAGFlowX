import pytest
from backend.app.evaluation.metrics.retrieval import (
    compute_precision_at_k, compute_recall_at_k, compute_mrr, compute_ndcg_at_k, compute_hit_rate_at_k, evaluate_retrieval
)
from backend.app.evaluation.metrics.generation import compute_token_f1, compute_exact_match, compute_answer_relevance

def test_retrieval_metrics():
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    relevant = {"doc2", "doc5"}

    # Precision@2 for retrieved ["doc1", "doc2"] -> 1 hit ("doc2") / 2 = 0.5
    assert compute_precision_at_k(retrieved, relevant, 2) == 0.5

    # Recall@2 -> 1 hit out of 2 relevant = 0.5
    assert compute_recall_at_k(retrieved, relevant, 2) == 0.5

    # MRR -> first relevant is doc2 at index 2 -> 1/2 = 0.5
    assert compute_mrr(retrieved, relevant) == 0.5

    # Hit Rate@3 -> doc2 is in top 3 -> 1.0
    assert compute_hit_rate_at_k(retrieved, relevant, 3) == 1.0

    # Hit Rate@1 -> doc1 is not in relevant -> 0.0
    assert compute_hit_rate_at_k(retrieved, relevant, 1) == 0.0

def test_generation_metrics():
    pred = "IBM Quantum System Two"
    gold = "IBM Quantum System Two"
    assert compute_exact_match(pred, gold) is True
    assert compute_token_f1(pred, gold) == 1.0

    rel = compute_answer_relevance("What platform uses superconducting qubits?", "IBM Quantum System Two uses superconducting qubits.")
    assert rel > 0.0
