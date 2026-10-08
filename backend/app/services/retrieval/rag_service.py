from typing import List, Optional
from backend.app.config import settings
from backend.app.models.search import VectorSearchResult
from backend.app.schemas.search import VectorSearchResultItem
from backend.app.schemas.rag import RAGQueryResponse
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

    def build_grounded_prompt(self, query: str, chunks: List[VectorSearchResult]) -> tuple[str, str]:
        """Formats strict grounded RAG prompt and system instructions."""
        system_instruction = (
            "You are a strictly grounded Retrieval-Augmented Generation (RAG) assistant. "
            "Your task is to answer the user's question relying ONLY on the provided document context passages below. "
            "Do NOT use outside knowledge or make assumptions not supported by the text. "
            "If the context passages do not contain enough information to answer the question, "
            "respond clearly: 'I could not find enough relevant information in the uploaded documents to answer your question.'"
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

    def query_rag(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None
    ) -> RAGQueryResponse:
        """Executes Basic RAG pipeline: Dense Retrieval -> Prompt Construction -> LLM Generation -> Grounded Answer."""
        # 1. Retrieve top_k relevant chunks via Dense Vector Search
        retrieved_chunks = self.vec_service.search_similar_chunks(
            query=query,
            top_k=top_k,
            document_id=document_id
        )

        # 2. Check if no chunks retrieved
        if not retrieved_chunks:
            fallback_answer = "I could not find enough relevant information in the uploaded documents to answer your question."
            return RAGQueryResponse(
                query=query,
                answer=fallback_answer,
                total_retrieved_chunks=0,
                llm_model=self.llm_provider.model_name,
                retrieved_chunks=[]
            )

        # 3. Build grounded prompt
        system_instruction, user_prompt = self.build_grounded_prompt(query, retrieved_chunks)

        # 4. Generate answer via LLM
        generated_answer = self.llm_provider.generate(user_prompt, system_instruction)

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

        return RAGQueryResponse(
            query=query,
            answer=generated_answer,
            total_retrieved_chunks=len(chunk_items),
            llm_model=self.llm_provider.model_name,
            retrieved_chunks=chunk_items
        )


rag_service = RAGService()
