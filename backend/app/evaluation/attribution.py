from typing import List, Set, Dict, Any, Optional

def classify_failure(
    relevant_ids: Set[str],
    candidate_pool_ids: List[str],
    final_retrieved_ids: List[str],
    generation_success: bool
) -> Optional[str]:
    if not relevant_ids:
        return None

    rel_set = set(relevant_ids)
    pool_set = set(candidate_pool_ids)
    final_set = set(final_retrieved_ids)

    # Check if gold chunk reached final retrieved set
    found_in_final = any(rid in final_set for rid in rel_set)
    if found_in_final and generation_success:
        return None  # No failure

    # Check if gold chunk was in candidate pool before reranking
    found_in_pool = any(rid in pool_set for rid in rel_set)

    if not found_in_pool:
        return "retrieval_miss"
    elif found_in_pool and not found_in_final:
        return "rerank_loss"
    elif found_in_final and not generation_success:
        return "generation_error"
    else:
        return "retrieval_miss"
