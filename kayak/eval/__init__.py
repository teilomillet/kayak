"""Evaluate decisions, rankings, and caller-owned RAG applications.

Example::

    from kayak.eval import banking77, evaluate, prepare_banking77
    prepare_banking77()
    suite = banking77(split="dev", limit=77)
    report = evaluate(model_or_client, suite, output=".benchmarks/eval/baseline")

Dataset downloads and inference are explicit. Importing this module needs only
Kayak's base dependencies. Full benchmark runs remain outside CI/CD.
Case and RAG callbacks own their backends; ranking and RAG scoring are pure.
"""

from ._banking77 import banking77, prepare_banking77
from ._benchmark import (
    BenchmarkComparison,
    BenchmarkResult,
    BenchmarkRun,
    benchmark,
    render_benchmark,
    write_benchmark,
)
from ._cases import (
    CaseReport,
    ChoiceCase,
    ChoiceDataset,
    Dataset,
    RankingCase,
    RankingDataset,
    choice_metrics,
    evaluate_cases,
    parse_cases,
)
from ._cases import summarize as summarize_cases
from ._compare import Comparison, compare, render_comparison
from ._comparison_cases import CaseComparison
from ._metrics import summarize
from ._predictions import Prediction, PredictionSet, export_predictions, predictions_from_report
from ._rag import (
    RAGAnswerReview,
    RAGAssessment,
    RAGContext,
    RAGJudgments,
    RAGReplay,
    RAGTrace,
    assess_rag,
)
from ._rag_experiment import (
    RAGAttempt,
    RAGCase,
    RAGCaseResult,
    RAGDataset,
    RAGEvalConfig,
    RAGFailure,
    RAGGate,
    RAGGateResult,
    RAGInput,
    RAGMetricSummary,
    RAGOutput,
    RAGReport,
    RAGReview,
    RAGReviewInput,
    RAGReviewRecord,
    RAGScore,
    RAGSummary,
)
from ._rag_reports import load_rag_report, render_rag_report, save_rag_report, score_rag
from ._rag_runner import aevaluate_rag, evaluate_rag
from ._ranking import RankedOutput, RankingAssessment, assess_ranking, ranking_metrics
from ._runner import DecisionBackend, evaluate, load_report
from ._schema import Example, Report, Suite
from ._scoring import (
    Metric,
    MetricInput,
    MetricResult,
    MetricSample,
    confidence_metrics,
    default_metrics,
    expected_calibration_error,
    log_loss,
    metric_input,
    reliability_bins,
    score_metrics,
    top_k_accuracy,
)
from ._suite import load_suite

__all__ = [
    "BenchmarkComparison",
    "BenchmarkResult",
    "BenchmarkRun",
    "CaseComparison",
    "CaseReport",
    "ChoiceCase",
    "ChoiceDataset",
    "Comparison",
    "Dataset",
    "DecisionBackend",
    "Example",
    "Metric",
    "MetricInput",
    "MetricResult",
    "MetricSample",
    "Prediction",
    "PredictionSet",
    "RAGAnswerReview",
    "RAGAssessment",
    "RAGAttempt",
    "RAGCase",
    "RAGCaseResult",
    "RAGContext",
    "RAGDataset",
    "RAGEvalConfig",
    "RAGFailure",
    "RAGGate",
    "RAGGateResult",
    "RAGInput",
    "RAGJudgments",
    "RAGMetricSummary",
    "RAGOutput",
    "RAGReplay",
    "RAGReport",
    "RAGReview",
    "RAGReviewInput",
    "RAGReviewRecord",
    "RAGScore",
    "RAGSummary",
    "RAGTrace",
    "RankedOutput",
    "RankingAssessment",
    "RankingCase",
    "RankingDataset",
    "Report",
    "Suite",
    "aevaluate_rag",
    "assess_rag",
    "assess_ranking",
    "banking77",
    "benchmark",
    "choice_metrics",
    "compare",
    "confidence_metrics",
    "default_metrics",
    "evaluate",
    "evaluate_cases",
    "evaluate_rag",
    "expected_calibration_error",
    "export_predictions",
    "load_rag_report",
    "load_report",
    "load_suite",
    "log_loss",
    "metric_input",
    "parse_cases",
    "predictions_from_report",
    "prepare_banking77",
    "ranking_metrics",
    "reliability_bins",
    "render_benchmark",
    "render_comparison",
    "render_rag_report",
    "save_rag_report",
    "score_rag",
    "score_metrics",
    "summarize",
    "summarize_cases",
    "top_k_accuracy",
    "write_benchmark",
]
