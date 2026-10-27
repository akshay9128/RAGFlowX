import math
from typing import List, Set, Dict, Any

def compute_precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if k <= 0 or not retrieved_ids:
        return 0.0
    k_retrieved = retrieved_ids[:k]
    hits = sum(1 for doc_id in k_retrieved if doc_id in relevant_ids)
    return hits / float(min(k, len(k_retrieved)))

def compute_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    if k <= 0 or not retrieved_ids:
        return 0.0
    k_retrieved = retrieved_ids[:k]
    hits = sum(1 for doc_id in k_retrieved if doc_id in relevant_ids)
    return hits / float(len(relevant_ids))

def compute_mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    if not relevant_ids or not retrieved_ids:
        return 0.0
    for idx, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / float(idx)
    return 0.0

def compute_ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0
    k_retrieved = retrieved_ids[:k]

    # DCG
    dcg = 0.0
    for idx, doc_id in enumerate(k_retrieved, start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(idx + 1)

    # IDCG
    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def compute_hit_rate_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0
    k_retrieved = retrieved_ids[:k]
    return 1.0 if any(doc_id in relevant_ids for doc_id in k_retrieved) else 0.0

def evaluate_retrieval(retrieved_ids: List[str], relevant_ids: Set[str], k_values: List[int] = [1, 3, 5, 10], candidate_pool_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    rel_set = set(relevant_ids)

    precisions = {}
    recalls = {}
    ndcgs = {}
    hit_rates = {}

    for k in k_values:
        precisions[k] = compute_precision_at_k(retrieved_ids, rel_set, k)
        recalls[k] = compute_recall_at_k(retrieved_ids, rel_set, k)
        ndcgs[k] = compute_ndcg_at_k(retrieved_ids, rel_set, k)
        hit_rates[k] = compute_hit_rate_at_k(retrieved_ids, rel_set, k)

    mrr = compute_mrr(retrieved_ids, rel_set)

    candidate_recall = 0.0
    if candidate_pool_ids is not None and rel_set:
        pool_set = set(candidate_pool_ids)
        hits = sum(1 for rid in rel_set if rid in pool_set)
        candidate_recall = hits / float(len(rel_set))

    return {
        "precision_at_k": precisions,
        "recall_at_k": recalls,
        "mrr": mrr,
        "ndcg_at_k": ndcgs,
        "hit_rate_at_k": hit_rates,
        "candidate_pool_recall": candidate_recall
    }
