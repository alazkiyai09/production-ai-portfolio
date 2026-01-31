# ============================================================
# Enterprise-RAG: Evaluation Routes
# ============================================================
"""
Evaluation endpoints for the RAG API.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.config import settings
from src.evaluation import EvaluationSample, RAGEvaluator, DEFAULT_TEST_SAMPLES
from src.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ============================================================
# Request/Response Models
# ============================================================

class EvaluationRequest(BaseModel):
    """Request model for evaluation."""

    samples: Optional[list[dict]] = Field(
        default=None,
        description="Custom evaluation samples (if None, uses default test set)",
    )
    num_samples: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of default samples to use (if samples not provided)",
    )


class EvaluationResponse(BaseModel):
    """Response model for evaluation results."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall_score: float
    evaluation_time: float
    num_samples: int


class SingleEvaluationRequest(BaseModel):
    """Request model for single question evaluation."""

    question: str = Field(..., min_length=1, max_length=1000)
    ground_truth: str = Field(..., min_length=1, max_length=2000)


class SingleEvaluationResponse(BaseModel):
    """Response model for single evaluation."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer: str


# ============================================================
# Routes
# ============================================================

@router.post("/run", response_model=EvaluationResponse)
async def run_evaluation(request: EvaluationRequest):
    """
    Run RAGAS evaluation on the RAG system.

    Evaluates the system on four metrics:
    - **Faithfulness**: Factual consistency with context
    - **Answer Relevancy**: Relevance to question
    - **Context Precision**: Retrieved context relevance
    - **Context Recall**: Ground truth coverage

    If no custom samples provided, uses the default test dataset.

    Args:
        request: Evaluation request with optional samples

    Returns:
        Evaluation results with all metric scores

    Example:
        POST /api/v1/evaluation/run
        {
            "samples": null,
            "num_samples": 10
        }
    """
    from fastapi import Request

    request_obj = Request.scope()["app"]
    rag_evaluator = getattr(request_obj.state, "rag_evaluator", None)

    if rag_evaluator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evaluator not initialized or evaluation is disabled",
        )

    try:
        logger.info("Starting RAGAS evaluation")

        # Get samples
        if request.samples is None:
            # Use default samples
            samples = DEFAULT_TEST_SAMPLES[: request.num_samples]
        else:
            # Use custom samples
            samples = [EvaluationSample(**s) for s in request.samples]

        # Run evaluation
        result = rag_evaluator.evaluate_samples(samples, show_progress=True)

        logger.info(f"Evaluation complete: {result.overall_score:.3f} overall")

        return EvaluationResponse(
            faithfulness=result.faithfulness,
            answer_relevancy=result.answer_relevancy,
            context_precision=result.context_precision,
            context_recall=result.context_recall,
            overall_score=result.overall_score,
            evaluation_time=result.evaluation_time,
            num_samples=result.num_samples,
        )

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}",
        )


@router.post("/evaluate-single", response_model=SingleEvaluationResponse)
async def evaluate_single(request: SingleEvaluationRequest):
    """
    Evaluate a single question-answer pair.

    Args:
        request: Single evaluation request

    Returns:
        Metric scores for the single question

    Example:
        POST /api/v1/evaluation/evaluate-single
        {
            "question": "What is the refund policy?",
            "ground_truth": "Refunds take 5-7 business days."
        }
    """
    from fastapi import Request

    request_obj = Request.scope()["app"]
    rag_chain = getattr(request_obj.state, "rag_chain", None)
    rag_evaluator = getattr(request_obj.state, "rag_evaluator", None)

    if rag_chain is None or rag_evaluator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG components not initialized",
        )

    try:
        # First generate answer
        response = rag_chain.query(request.question)

        # Then evaluate
        scores = rag_evaluator.evaluate_single(
            question=request.question,
            ground_truth=request.ground_truth,
        )

        return SingleEvaluationResponse(
            faithfulness=scores.get("faithfulness", 0.0),
            answer_relevancy=scores.get("answer_relevancy", 0.0),
            context_precision=scores.get("context_precision", 0.0),
            context_recall=scores.get("context_recall", 0.0),
            answer=response.answer,
        )

    except Exception as e:
        logger.error(f"Single evaluation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}",
        )


@router.get("/report")
async def generate_report():
    """
    Generate an evaluation report.

    Runs evaluation and returns a markdown report.

    Example:
        GET /api/v1/evaluation/report
    """
    from fastapi import Response
    from fastapi import Request

    request_obj = Request.scope()["app"]
    rag_evaluator = getattr(request_obj.state, "rag_evaluator", None)

    if rag_evaluator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evaluator not initialized",
        )

    try:
        # Run evaluation on default samples
        result = rag_evaluator.evaluate_samples(DEFAULT_TEST_SAMPLES[:5])

        # Generate report
        report = rag_evaluator.generate_report(result)

        return Response(content=report, media_type="text/markdown")

    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate report: {str(e)}",
        )


@router.get("/metrics")
async def get_available_metrics():
    """
    Get list of available evaluation metrics.

    Example:
        GET /api/v1/evaluation/metrics
    """
    return {
        "metrics": {
            "faithfulness": {
                "name": "Faithfulness",
                "description": "Factual consistency of generated answer with retrieved context",
                "range": "0-1",
                "higher_is_better": True,
            },
            "answer_relevancy": {
                "name": "Answer Relevancy",
                "description": "Relevance of generated answer to the question",
                "range": "0-1",
                "higher_is_better": True,
            },
            "context_precision": {
                "name": "Context Precision",
                "description": "Signal-to-noise ratio in retrieved contexts",
                "range": "0-1",
                "higher_is_better": True,
            },
            "context_recall": {
                "name": "Context Recall",
                "description": "Coverage of ground truth by retrieved contexts",
                "range": "0-1",
                "higher_is_better": True,
            },
        },
        "enabled": settings.ENABLE_EVALUATION,
    }
