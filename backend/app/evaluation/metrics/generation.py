import re
from typing import List, Dict, Any, Optional
from backend.app.services.citations.service import verification_service
from backend.app.models.search import VectorSearchResult

def compute_token_f1(prediction: str, gold: str) -> float:
    pred_tokens = re.findall(r'\w+', prediction.lower())
    gold_tokens = re.findall(r'\w+', gold.lower())

    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    num_same = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_exact_match(prediction: str, gold: str) -> bool:
    def normalize(s: str) -> str:
        return " ".join(s.lower().split())
    return normalize(prediction) == normalize(gold)

def compute_answer_relevance(question: str, answer: str) -> float:
    q_tokens = set(re.findall(r'\w+', question.lower()))
    a_tokens = set(re.findall(r'\w+', answer.lower()))
    if not q_tokens:
        return 0.0
    common = q_tokens & a_tokens
    return len(common) / len(q_tokens)

def evaluate_generation(
    question: str,
    generated_answer: str,
    gold_answer: str,
    answerable: bool,
    retrieved_chunks: List[Dict[str, Any]],
    confidence_score: float = 1.0
) -> Dict[str, Any]:
    # Groundedness via VerificationService
    vec_search_results = [
        VectorSearchResult(
            chunk_id=c.get("chunk_id", "c1"),
            document_id=c.get("document_id", "doc_id"),
            filename=c.get("filename", "doc.txt"),
            page_number=c.get("page_number", 1),
            chunk_index=c.get("chunk_index", 0),
            text=c.get("text", ""),
            score=c.get("score", 1.0),
            source=c.get("source", "doc.txt")
        ) for c in retrieved_chunks
    ]
    _, groundedness_score, unsupported, _ = verification_service.verify_answer(generated_answer, vec_search_results)
    total_claims = len(verification_service.extract_claims(generated_answer))
    citation_accuracy = 1.0 if not unsupported else max(0.0, 1.0 - (len(unsupported) / (total_claims or 1)))

    answer_relevance = compute_answer_relevance(question, generated_answer)
    token_f1 = compute_token_f1(generated_answer, gold_answer)
    exact_match = compute_exact_match(generated_answer, gold_answer)

    # Abstention metrics
    correct_abstention = None
    if not answerable:
        # Check if the system correctly abstained
        abstention_phrases = ["not in", "cannot answer", "does not contain", "not mentioned", "don't know", "unable"]
        is_abstaining = any(phrase in generated_answer.lower() for phrase in abstention_phrases)
        correct_abstention = is_abstaining

    return {
        "groundedness_score": groundedness_score,
        "answer_relevance": answer_relevance,
        "citation_accuracy": citation_accuracy,
        "token_f1": token_f1,
        "exact_match": exact_match,
        "correct_abstention": correct_abstention,
        "confidence_score": confidence_score
    }
