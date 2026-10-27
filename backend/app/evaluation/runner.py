import uuid
import datetime
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from rank_bm25 import BM25Okapi

from backend.app.evaluation.schemas import (
    BenchmarkDataset, AblationConfig, RunReport, QuestionEvaluationResult,
    RetrievalMetricsResult, GenerationMetricsResult, LatencyMetricsResult, RunComparisonResult
)
from backend.app.evaluation.metrics.retrieval import evaluate_retrieval
from backend.app.evaluation.metrics.generation import evaluate_generation
from backend.app.evaluation.metrics.latency import LatencyTracker
from backend.app.evaluation.attribution import classify_failure
from backend.app.evaluation.store import EvaluationStore

# Pipeline services
from backend.app.services.chunking.splitter import TextSplitter
from backend.app.services.embeddings.factory import get_embedding_provider
from backend.app.services.embeddings.service import EmbeddingService
from backend.app.database.vector_store.in_memory import InMemoryVectorStore
from backend.app.services.retrieval.vector_service import VectorSearchService
from backend.app.services.retrieval.bm25_service import BM25SearchService
from backend.app.services.retrieval.hybrid_service import HybridSearchService
from backend.app.services.reranking.factory import get_rerank_provider
from backend.app.services.reranking.service import RerankService
from backend.app.services.generation.factory import get_llm_provider
from backend.app.services.retrieval.rag_service import RAGService
from backend.app.models.search import VectorSearchResult
from backend.app.models.chunk import ChunkingConfig, DocumentChunk, ChunkMetadata
from backend.app.models.embedding import ChunkEmbedding

class EvaluationRunner:
    def __init__(self, store: Optional[EvaluationStore] = None):
        self.store = store or EvaluationStore()

    def run_benchmark(
        self,
        dataset: BenchmarkDataset,
        config: AblationConfig,
        split: str = "dev",
        git_commit: Optional[str] = None
    ) -> RunReport:
        run_id = str(uuid.uuid4())[:8]
        timestamp = datetime.datetime.utcnow().isoformat()

        # Filter items by split
        items = [item for item in dataset.items if item.split == split]

        # 1. Setup isolated temporary index and services for this ablation run
        vector_store = InMemoryVectorStore()
        embed_provider = get_embedding_provider()
        embedding_service = EmbeddingService(embed_provider, vector_store)
        vector_search_service = VectorSearchService(embed_service=embedding_service, vec_store=vector_store)
        bm25_service = BM25SearchService()
        reranker_provider = get_rerank_provider()
        rerank_service = RerankService(reranker_provider)
        hybrid_search_service = HybridSearchService(vector_search_service, bm25_service)

        llm_provider = get_llm_provider()
        rag_service = RAGService(vector_search_service, llm_provider)

        # Ingest documents into isolated indexes
        chunk_map: Dict[str, str] = {}  # chunk_id -> chunk text mapping
        for doc_def in dataset.documents:
            filename = doc_def.get("filename", "doc.txt")
            content = doc_def.get("content", "")

            # Chunk text
            chunk_size = config.chunk_size or 500
            chunk_overlap = config.chunk_overlap or 50
            splitter = TextSplitter(ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap))
            chunks_with_offsets = splitter.split_text(content)
            chunks = [c[0] for c in chunks_with_offsets]

            # Index into vector store and BM25
            doc_id = str(uuid.uuid4())[:8]
            chunk_dicts = []
            for idx, c_text in enumerate(chunks):
                chunk_id = f"chunk_{doc_id}_{idx}"
                chunk_map[chunk_id] = c_text
                chunk_dicts.append({
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "filename": filename,
                    "page_number": 1,
                    "chunk_index": idx,
                    "text": c_text
                })

            # Embed and index vector store
            embed_texts = [cd["text"] for cd in chunk_dicts]
            embeddings = embed_provider.embed_documents(embed_texts)
            chunk_embeddings = []
            for cd, emb in zip(chunk_dicts, embeddings):
                chunk_obj = DocumentChunk(
                    text=cd["text"],
                    metadata=ChunkMetadata(
                        chunk_id=cd["chunk_id"],
                        document_id=cd["document_id"],
                        filename=cd["filename"],
                        page_number=cd["page_number"],
                        chunk_index=cd["chunk_index"],
                        source=cd["filename"],
                        character_count=len(cd["text"])
                    )
                )
                chunk_embeddings.append(
                    ChunkEmbedding(
                        chunk=chunk_obj,
                        embedding=emb,
                        embedding_model=embed_provider.model_name,
                        dimension=embed_provider.dimension
                    )
                )
            vector_store.add_embeddings(chunk_embeddings)

            # Index BM25
            bm25_chunks = [
                DocumentChunk(
                    text=cd["text"],
                    metadata=ChunkMetadata(
                        chunk_id=cd["chunk_id"],
                        document_id=cd["document_id"],
                        filename=cd["filename"],
                        page_number=cd["page_number"],
                        chunk_index=cd["chunk_index"],
                        source=cd["filename"],
                        character_count=len(cd["text"])
                    )
                ) for cd in chunk_dicts
            ]
            bm25_service._indexed_chunks[doc_id] = bm25_chunks
            tokenized_chunks = [bm25_service._tokenize(c.text) for c in bm25_chunks]
            bm25_service._bm25_indices[doc_id] = BM25Okapi(tokenized_chunks)

        question_results: List[QuestionEvaluationResult] = []
        failure_counts = {"retrieval_miss": 0, "rerank_loss": 0, "generation_error": 0}

        tracker = LatencyTracker()

        for item in items:
            # Resolve relevant chunks matching document filename or content match
            relevant_ids = set()
            for cid, ctext in chunk_map.items():
                if item.document_filename and item.document_filename in cid or (item.gold_answer.lower() in ctext.lower()):
                    relevant_ids.add(cid)
            if not relevant_ids and chunk_map:
                relevant_ids = {list(chunk_map.keys())[0]}

            # Execute retrieval based on mode
            top_k = config.top_k or 3
            candidate_pool_size = config.candidate_pool_size or 10
            mode = config.retrieval_mode

            with tracker.measure("retrieval"):
                if mode == "dense":
                    q_emb = embed_provider.embed_query(item.question)
                    search_res = vector_store.similarity_search(q_emb, top_k=top_k)
                    retrieved_chunks = [{"chunk_id": r.chunk_id, "text": r.text, "score": r.score, "source": r.source} for r in search_res]
                    candidate_pool_ids = [r.chunk_id for r in search_res]
                elif mode == "bm25":
                    bm25_res = bm25_service.search(item.question, top_k=top_k)
                    retrieved_chunks = [{"chunk_id": r.chunk_id, "text": r.text, "score": r.score, "source": r.source} for r in bm25_res]
                    candidate_pool_ids = [r.chunk_id for r in bm25_res]
                else:
                    # Hybrid
                    hybrid_res = hybrid_search_service.search(
                        query=item.question,
                        top_k=candidate_pool_size
                    )
                    candidate_pool_ids = [r.chunk_id for r in hybrid_res]

                    if mode == "hybrid_rerank":
                        # Convert SearchResult objects to dicts for reranking
                        search_res_objs = hybrid_res
                        reranked = rerank_service.rerank_results(item.question, search_res_objs, top_k=top_k)
                        retrieved_chunks = [{"chunk_id": r.chunk_id, "text": r.text, "score": r.score, "source": r.source} for r in reranked]
                    else:
                        retrieved_chunks = [{"chunk_id": r.chunk_id, "text": r.text, "score": r.score, "source": r.source} for r in hybrid_res[:top_k]]

            retrieved_ids = [c["chunk_id"] for c in retrieved_chunks]

            # Generation stage
            with tracker.measure("generation"):
                vec_search_results = [
                    VectorSearchResult(
                        chunk_id=c["chunk_id"],
                        document_id="doc_id",
                        filename="doc.txt",
                        page_number=1,
                        chunk_index=0,
                        text=c["text"],
                        score=c["score"],
                        source=c.get("source", "doc.txt")
                    ) for c in retrieved_chunks
                ]
                sys_inst, user_prompt = rag_service.build_grounded_prompt(item.question, vec_search_results)
                gen_answer = llm_provider.generate(user_prompt, sys_inst)
                confidence = 0.95

            # Evaluate retrieval
            retrieval_eval = evaluate_retrieval(retrieved_ids, relevant_ids, k_values=[1, 3, 5, 10], candidate_pool_ids=candidate_pool_ids)

            # Evaluate generation
            gen_eval = evaluate_generation(
                question=item.question,
                generated_answer=gen_answer,
                gold_answer=item.gold_answer,
                answerable=item.answerable,
                retrieved_chunks=retrieved_chunks,
                confidence_score=confidence
            )

            # Failure attribution
            gen_success = gen_eval["token_f1"] > 0.2 or gen_eval["exact_match"]
            failure_cause = classify_failure(relevant_ids, candidate_pool_ids, retrieved_ids, gen_success)
            if failure_cause:
                failure_counts[failure_cause] = failure_counts.get(failure_cause, 0) + 1

            q_result = QuestionEvaluationResult(
                item_id=item.id,
                question=item.question,
                retrieved_chunks=retrieved_chunks,
                generated_answer=gen_answer,
                gold_answer=item.gold_answer,
                answerable=item.answerable,
                retrieval_metrics=RetrievalMetricsResult(**retrieval_eval),
                generation_metrics=GenerationMetricsResult(**gen_eval),
                latency=LatencyMetricsResult(stages=tracker.summarize()),
                failure_cause=failure_cause
            )
            question_results.append(q_result)

        # Compute aggregate metrics across items
        agg_metrics = {}
        if question_results:
            agg_metrics["mean_mrr"] = float(np.mean([q.retrieval_metrics.mrr for q in question_results]))
            agg_metrics["mean_precision_at_3"] = float(np.mean([q.retrieval_metrics.precision_at_k.get(3, 0.0) for q in question_results]))
            agg_metrics["mean_recall_at_3"] = float(np.mean([q.retrieval_metrics.recall_at_k.get(3, 0.0) for q in question_results]))
            agg_metrics["mean_groundedness"] = float(np.mean([q.generation_metrics.groundedness_score for q in question_results]))
            agg_metrics["mean_token_f1"] = float(np.mean([q.generation_metrics.token_f1 for q in question_results]))
            agg_metrics["exact_match_rate"] = float(np.mean([1.0 if q.generation_metrics.exact_match else 0.0 for q in question_results]))

        report = RunReport(
            run_id=run_id,
            timestamp=timestamp,
            git_commit=git_commit,
            dataset_name=dataset.name,
            split=split,
            config=config,
            aggregate_metrics=agg_metrics,
            failure_attribution_summary=failure_counts,
            latency_summary=tracker.summarize(),
            question_results=question_results
        )

        return self.store.save_run(report)

    def compare_runs(self, run_id_a: str, run_id_b: str, threshold: float = 0.05) -> RunComparisonResult:
        run_a = self.store.get_run(run_id_a)
        run_b = self.store.get_run(run_id_b)

        if not run_a or not run_b:
            raise ValueError(f"Run IDs not found: {run_id_a}, {run_id_b}")

        deltas = {}
        regressions = []
        noise_band = []
        passed = True

        keys = set(run_a.aggregate_metrics.keys()) | set(run_b.aggregate_metrics.keys())
        for k in keys:
            val_a = run_a.aggregate_metrics.get(k, 0.0)
            val_b = run_b.aggregate_metrics.get(k, 0.0)
            delta = val_b - val_a
            deltas[k] = delta

            if delta < -threshold:
                regressions.append(k)
                passed = False
            elif abs(delta) < 0.01:
                noise_band.append(k)

        return RunComparisonResult(
            run_id_a=run_id_a,
            run_id_b=run_id_b,
            deltas=deltas,
            regressions_detected=regressions,
            within_noise_band=noise_band,
            passed=passed
        )

    def generate_markdown_report(self, report: RunReport) -> str:
        lines = [
            f"# RAGFlowX Evaluation Report: {report.run_id}",
            f"- **Dataset:** {report.dataset_name} (Split: {report.split})",
            f"- **Timestamp:** {report.timestamp}",
            f"- **Retrieval Mode:** {report.config.retrieval_mode}",
            f"- **Chunk Size:** {report.config.chunk_size or 'Default'} | **Overlap:** {report.config.chunk_overlap or 'Default'}",
            "",
            "## Aggregate Metrics",
            "| Metric | Value |",
            "|---|---|",
        ]
        for k, v in report.aggregate_metrics.items():
            lines.append(f"| {k} | {v:.4f} |")

        lines.extend([
            "",
            "## Failure Attribution",
        ])
        for cause, count in report.failure_attribution_summary.items():
            lines.append(f"- **{cause}:** {count}")

        return "\n".join(lines)
