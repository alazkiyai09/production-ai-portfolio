# ============================================================
# Enterprise-RAG: Evaluation Module
# ============================================================
"""
RAG evaluation using RAGAS metrics.

This module provides:
- RAGAS metric integration
- Test dataset management
- Batch evaluation
- Evaluation reports
"""

from src.evaluation.rag_evaluator import (
    EvaluationResult,
    EvaluationSample,
    RAGEvaluator,
)

__all__ = [
    "RAGEvaluator",
    "EvaluationSample",
    "EvaluationResult",
]
