import re
from typing import List, Tuple, Optional
from backend.app.models.search import VectorSearchResult
from backend.app.schemas.rag import Citation
from backend.app.schemas.search import VectorSearchResultItem
from backend.app.schemas.verification import VerifiedClaim, VerifiedAnswerResponse, StandaloneVerificationResponse


class VerificationService:
    """Service for extracting claims from generated answers and verifying them against retrieved context chunks."""

    def extract_claims(self, answer_text: str) -> List[str]:
        """Split generated answer text into atomic claims (sentences/statements)."""
        if not answer_text or not answer_text.strip():
            return []
        # Split by sentence boundaries (.?! or newlines)
        raw_sentences = re.split(r'(?<=[.?!])\s+|\n+', answer_text.strip())
        claims = [s.strip() for s in raw_sentences if s.strip() and len(s.strip()) > 3]
        if not claims and answer_text.strip():
            claims = [answer_text.strip()]
        return claims

    def compute_claim_chunk_overlap(self, claim: str, chunk_text: str) -> float:
        """Compute lexical word overlap (Jaccard similarity) between a claim and a chunk text."""
        claim_words = set(re.findall(r'\w+', claim.lower()))
        chunk_words = set(re.findall(r'\w+', chunk_text.lower()))
        if not claim_words or not chunk_words:
            return 0.0
        intersection = claim_words.intersection(chunk_words)
        union = claim_words.union(chunk_words)
        jaccard = len(intersection) / len(union) if union else 0.0

        # Boost if key terms or substring match
        sub_match = 0.3 if claim.lower() in chunk_text.lower() else 0.0

        # Combined heuristic score
        score = min(1.0, (jaccard * 2.0) + sub_match)
        return round(score, 4)

    def verify_claim(self, claim: str, chunks: List[VectorSearchResult]) -> VerifiedClaim:
        """Verify a single claim against a list of retrieved chunks."""
        if not chunks:
            return VerifiedClaim(
                claim_text=claim,
                support_status="unsupported",
                support_score=0.0,
                supporting_chunk_id=None,
                supporting_citation=None,
                reason="No context chunks available for verification."
            )

        best_score = 0.0
        best_chunk = None

        for chunk in chunks:
            score = self.compute_claim_chunk_overlap(claim, chunk.text)
            if score > best_score:
                best_score = score
                best_chunk = chunk

        # Determine support status based on score thresholds
        if best_score >= 0.35 and best_chunk is not None:
            status = "supported"
            reason = f"Claim supported by chunk {best_chunk.chunk_index} from {best_chunk.filename} (score: {best_score})"
        elif 0.15 <= best_score < 0.35 and best_chunk is not None:
            status = "partially_supported"
            reason = f"Claim partially supported by chunk {best_chunk.chunk_index} from {best_chunk.filename} (score: {best_score})"
        else:
            status = "unsupported"
            reason = "No sufficient textual evidence found in retrieved chunks."

        citation = None
        if best_chunk is not None and status != "unsupported":
            citation = Citation(
                document_id=best_chunk.document_id,
                filename=best_chunk.filename,
                page_number=best_chunk.page_number,
                chunk_index=best_chunk.chunk_index,
                quoted_text=best_chunk.text
            )

        return VerifiedClaim(
            claim_text=claim,
            support_status=status,
            support_score=best_score,
            supporting_chunk_id=best_chunk.chunk_id if best_chunk else None,
            supporting_citation=citation,
            reason=reason
        )

    def verify_answer(
        self,
        answer: str,
        chunks: List[VectorSearchResult]
    ) -> Tuple[List[VerifiedClaim], float, List[str], str]:
        """Verify all atomic claims in an answer against chunks.

        Returns:
            - list of VerifiedClaim
            - overall groundedness score (0.0 to 1.0)
            - list of unsupported claim texts
            - verified/annotated answer string
        """
        claims = self.extract_claims(answer)
        if not claims:
            return [], 0.0, [], answer

        verified_claims = [self.verify_claim(c, chunks) for c in claims]

        total_score = sum(vc.support_score for vc in verified_claims)
        groundedness = round(total_score / len(verified_claims), 4) if verified_claims else 0.0

        unsupported = [vc.claim_text for vc in verified_claims if vc.support_status == "unsupported"]

        # Build annotated verified answer
        annotated_sentences = []
        for vc in verified_claims:
            if vc.support_status == "supported" and vc.supporting_citation:
                cit = vc.supporting_citation
                annotated_sentences.append(f"{vc.claim_text} [Source: {cit.filename} p.{cit.page_number}]")
            elif vc.support_status == "partially_supported" and vc.supporting_citation:
                cit = vc.supporting_citation
                annotated_sentences.append(f"{vc.claim_text} [Partial Source: {cit.filename} p.{cit.page_number}]")
            else:
                annotated_sentences.append(f"{vc.claim_text} [Unverified]")

        verified_answer_text = " ".join(annotated_sentences)

        return verified_claims, groundedness, unsupported, verified_answer_text


verification_service = VerificationService()
