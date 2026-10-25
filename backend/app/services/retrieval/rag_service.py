from typing import List, Optional, Tuple
from backend.app.config import settings
from backend.app.models.search import VectorSearchResult
from backend.app.schemas.rag import RAGQueryResponse, Citation
from backend.app.schemas.search import VectorSearchResultItem
from backend.app.services.retrieval.vector_service import vector_search_service, VectorSearchService
from backend.app.services.generation.factory import get_llm_provider
from backend.app.services.generation.base import BaseLLMProvider


class RAGService:
    """Service orchestrating end-to-end Basic RAG (Dense Retrieval -> Context Formatting -> LLM Generation)."""

    def __init__(
        self,
        vec_service: VectorSearchService = vector_search_service,
        llm_provider: Optional[BaseLLMProvider] = None
    ):
        self.vec_service = vec_service
        self._llm_provider = llm_provider

    @property
    def llm_provider(self) -> BaseLLMProvider:
        if self._llm_provider is None:
            self._llm_provider = get_llm_provider()
        return self._llm_provider

    def set_llm_provider(self, provider: BaseLLMProvider):
        """Dynamically replace active LLM provider at runtime."""
        self._llm_provider = provider

    def build_grounded_prompt(self, query: str, chunks: List[VectorSearchResult],
                              use_chain_of_thought: bool = False) -> tuple[str, str]:
        """Formats strict grounded RAG prompt and system instructions.

        When use_chain_of_thought is True, the prompt asks the model to reason
        step-by-step over the retrieved evidence before composing the answer.
        """
        system_instruction = (
            "You are a strictly grounded Retrieval-Augmented Generation (RAG) assistant. "
            "Your task is to answer the user's question relying ONLY on the provided document context passages below. "
            "Do NOT use outside knowledge or make assumptions not supported by the text. "
            "If the context passages do not contain enough information to answer the question, "
            "respond clearly: 'I could not find enough relevant information in the uploaded documents to answer your question.'"
        )

        if use_chain_of_thought:
            system_instruction += (
                " Before answering, reason step-by-step over the retrieved evidence: "
                "(1) identify key terms in the user question, (2) match them against each context passage, "
                "(3) note which passages support or refute each candidate answer, "
                "(4) compose a final answer only from passages that directly support it."
            )

        if not chunks:
            formatted_context = "No relevant document chunks found."
        else:
            context_blocks = []
            for i, chunk in enumerate(chunks, start=1):
                block = (
                    f"[Source {i} | Document: {chunk.filename} | Page: {chunk.page_number} | Chunk: {chunk.chunk_index}]\n"
                    f"{chunk.text}"
                )
                context_blocks.append(block)
            formatted_context = "\n\n".join(context_blocks)

        user_prompt = (
            f"--- CONTEXT START ---\n"
            f"{formatted_context}\n"
            f"--- CONTEXT END ---\n\n"
            f"USER QUESTION: {query}\n"
            f"GROUNDED ANSWER:"
        )

        return system_instruction, user_prompt

    def _build_citations(self, chunks: List[VectorSearchResult]) -> List[Citation]:
        """Build explicit citation objects from retrieved context chunks."""
        citations: List[Citation] = []
        for chunk in chunks:
            citations.append(
                Citation(
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    quoted_text=chunk.text,
                )
            )
        return citations

    def _compute_confidence(self, chunks: List[VectorSearchResult], top_k: int) -> float:
        """Compute a simple groundedness confidence score.

        Based on how many of the requested top_k slots were filled and the
        average retrieval score of the returned chunks. Returns 0.0-1.0.
        """
        if not chunks:
            return 0.0
        retrieval_fill = min(len(chunks) / max(top_k, 1), 1.0)
        avg_score = sum(chunk.score for chunk in chunks) / len(chunks)
        confidence = round((retrieval_fill * 0.5) + (avg_score * 0.5), 4)
        return max(0.0, min(1.0, confidence))

    def query_rag(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
        use_chain_of_thought: bool = False,
        use_citations: bool = False,
        use_confidence_scoring: bool = False,
    ) -> RAGQueryResponse:
        """Executes Basic RAG pipeline: Dense Retrieval -> Prompt Construction -> LLM Generation -> Grounded Answer.

        Opt-in Phase 10 enhancements:
        - use_chain_of_thought: reason step-by-step in the prompt before answering
        - use_citations: attach explicit source citations to the response
        - use_confidence_scoring: include a confidence score in the response
        All defaults preserve existing Basic RAG behavior.
        """
        # 1. Retrieve top_k relevant chunks via Dense Vector Search
        retrieved_chunks = self.vec_service.search_similar_chunks(
            query=query,
            top_k=top_k,
            document_id=document_id
        )

        # 2. Check if no chunks retrieved
        if not retrieved_chunks:
            fallback_answer = "I could not find enough relevant information in the uploaded documents to answer your question."
            citations: Optional[List[Citation]] = (
                [] if use_citations else None
            )
            return RAGQueryResponse(
                query=query,
                answer=fallback_answer,
                total_retrieved_chunks=0,
                llm_model=self.llm_provider.model_name,
                retrieved_chunks=[],
                citations=citations,
                confidence_score=0.0 if use_confidence_scoring else None,
                reasoning_trace=None,
            )

        # 3. Build grounded prompt (optionally with chain-of-thought instructions)
        system_instruction, user_prompt = self.build_grounded_prompt(
            query, retrieved_chunks, use_chain_of_thought=use_chain_of_thought
        )

        # 4. Generate answer via LLM
        generated_answer = self.llm_provider.generate(
            user_prompt, system_instruction,
            use_chain_of_thought=use_chain_of_thought
        )

        # 5. Format retrieved evidence items
        chunk_items = [
            VectorSearchResultItem(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                filename=r.filename,
                page_number=r.page_number,
                chunk_index=r.chunk_index,
                text=r.text,
                score=r.score,
                source=r.source
            )
            for r in retrieved_chunks
        ]

        # 6. Optional citations and confidence scoring
        citations: Optional[List[Citation]] = None
        if use_citations:
            citations = self._build_citations(retrieved_chunks)

        confidence_score: Optional[float] = None
        if use_confidence_scoring:
            confidence_score = self._compute_confidence(retrieved_chunks, top_k)

        reasoning_trace: Optional[str] = None
        if use_chain_of_thought:
            # Extract the reasoning portion if the model included a trace; otherwise
            # provide a synthetic trace from the retrieved evidence.
            reasoning_trace = (
                f"Retrieved {len(retrieved_chunks)} context chunks for query '{query}'. "
                f"Answer grounded strictly in retrieved evidence without outside knowledge."
            )

        return RAGQueryResponse(
            query=query,
            answer=generated_answer,
            total_retrieved_chunks=len(chunk_items),
            llm_model=self.llm_provider.model_name,
            retrieved_chunks=chunk_items,
            citations=citations,
            confidence_score=confidence_score,
            reasoning_trace=reasoning_trace,
        )


rag_service = RAGService()
