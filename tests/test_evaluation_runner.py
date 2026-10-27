import pytest
from backend.app.evaluation.datasets.loader import load_dataset
from backend.app.evaluation.schemas import AblationConfig
from backend.app.evaluation.runner import EvaluationRunner
from backend.app.evaluation.store import EvaluationStore
from pathlib import Path

def test_evaluation_runner_end_to_end(tmp_path):
    dataset_path = Path("backend/app/evaluation/datasets/gold_dataset.json")
    dataset = load_dataset(dataset_path)

    store = EvaluationStore(data_dir=tmp_path)
    runner = EvaluationRunner(store=store)

    config = AblationConfig(
        chunk_size=300,
        chunk_overlap=30,
        retrieval_mode="hybrid_rerank"
    )

    report = runner.run_benchmark(dataset, config, split="dev")
    assert report.run_id is not None
    assert len(report.question_results) > 0
    assert "mean_mrr" in report.aggregate_metrics

    # Test markdown generation
    md = runner.generate_markdown_report(report)
    assert "# RAGFlowX Evaluation Report" in md

    # Test comparison
    run_id_a = report.run_id
    report_b = runner.run_benchmark(dataset, config, split="dev")
    run_id_b = report_b.run_id

    comparison = runner.compare_runs(run_id_a, run_id_b)
    assert comparison.passed is True
