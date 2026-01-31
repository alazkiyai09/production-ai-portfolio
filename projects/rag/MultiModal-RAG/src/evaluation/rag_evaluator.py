# ============================================================
# Enterprise-RAG: RAG Evaluator
# ============================================================
"""
RAG evaluation using RAGAS metrics.

This module provides comprehensive evaluation of RAG systems using:
- Faithfulness: Factual consistency of answer with context
- Answer Relevancy: Relevance of answer to question
- Context Precision: Relevance of retrieved contexts
- Context Recall: Coverage of ground truth by contexts

Example:
    >>> from src.evaluation import RAGEvaluator, EvaluationSample
    >>> evaluator = RAGEvaluator(rag_chain)
    >>> samples = [
    ...     EvaluationSample(
    ...         question="What is the refund policy?",
    ...         ground_truth="Refunds are processed within 5-7 business days."
    ...     )
    ... ]
    >>> result = evaluator.evaluate_samples(samples)
    >>> print(result.overall_score)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.config import settings
from src.logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)


# ============================================================
# Data Classes
# ============================================================

@dataclass
class EvaluationSample:
    """
    Single sample for RAG evaluation.

    Attributes:
        question: Question to evaluate
        ground_truth: Reference answer for comparison
        contexts: Retrieved contexts (optional, generated if None)
        answer: Generated answer (optional, generated if None)
        metadata: Additional metadata for the sample

    Example:
        >>> sample = EvaluationSample(
        ...     question="What is the refund policy?",
        ...     ground_truth="Refunds are processed within 5-7 business days.",
        ...     metadata={"category": "policy"}
        ... )
    """

    question: str
    ground_truth: str
    contexts: Optional[list[str]] = None
    answer: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "contexts": self.contexts,
            "answer": self.answer,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationSample":
        """Create from dictionary."""
        return cls(
            question=data["question"],
            ground_truth=data["ground_truth"],
            contexts=data.get("contexts"),
            answer=data.get("answer"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EvaluationResult:
    """
    Complete evaluation result from RAGAS.

    Attributes:
        faithfulness: Factual consistency score (0-1)
        answer_relevancy: Relevance to question score (0-1)
        context_precision: Retrieved context relevance score (0-1)
        context_recall: Ground truth coverage score (0-1)
        overall_score: Average of all metrics (0-1)
        individual_results: Per-sample results
        evaluation_time: Time taken for evaluation
        num_samples: Number of samples evaluated

    Example:
        >>> result = evaluator.evaluate_samples(samples)
        >>> print(f"Overall: {result.overall_score:.3f}")
        >>> print(f"Faithfulness: {result.faithfulness:.3f}")
    """

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall_score: float
    individual_results: list[dict[str, Any]]
    evaluation_time: float
    num_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "overall_score": self.overall_score,
            "individual_results": self.individual_results,
            "evaluation_time": self.evaluation_time,
            "num_samples": self.num_samples,
        }


# ============================================================
# Default Test Dataset
# ============================================================

DEFAULT_TEST_SAMPLES = [
    EvaluationSample(
        question="What is the company's refund policy?",
        ground_truth="Refunds are processed within 5-7 business days for digital products and 10-14 days for physical products. Customers can request refunds through the support portal or by contacting customer service.",
        metadata={"category": "policy", "difficulty": "easy"},
    ),
    EvaluationSample(
        question="How do I cancel my subscription?",
        ground_truth="Subscriptions can be cancelled through the account settings page. For annual plans, cancellations take effect at the end of the billing cycle. Monthly plans can be cancelled immediately with a prorated refund.",
        metadata={"category": "subscription", "difficulty": "easy"},
    ),
    EvaluationSample(
        question="What security measures does the platform use?",
        ground_truth="The platform uses AES-256 encryption for data at rest, TLS 1.3 for data in transit, multi-factor authentication, and regular security audits. Data is stored in SOC 2 compliant data centers.",
        metadata={"category": "security", "difficulty": "medium"},
    ),
    EvaluationSample(
        question="What are the pricing tiers for the service?",
        ground_truth="The service offers three tiers: Basic ($9/month), Professional ($29/month), and Enterprise (custom pricing). Each tier includes different features and support levels.",
        metadata={"category": "pricing", "difficulty": "easy"},
    ),
    EvaluationSample(
        question="How is user data handled according to the privacy policy?",
        ground_truth="User data is collected for service provision and improvement. Data is not sold to third parties. Users can request data deletion. The policy complies with GDPR and CCPA regulations.",
        metadata={"category": "privacy", "difficulty": "medium"},
    ),
    EvaluationSample(
        question="What support channels are available to customers?",
        ground_truth="Support is available via email (support@example.com), live chat (24/7 for Professional and Enterprise), phone (business hours only), and a comprehensive knowledge base.",
        metadata={"category": "support", "difficulty": "easy"},
    ),
    EvaluationSample(
        question="What is the uptime guarantee?",
        ground_truth="The platform guarantees 99.9% uptime for Professional plans and 99.99% for Enterprise plans. Service credits are provided if uptime falls below the guaranteed level.",
        metadata={"category": "technical", "difficulty": "medium"},
    ),
    EvaluationSample(
        question="How does the API rate limiting work?",
        ground_truth="API rate limits are 100 requests/minute for Basic, 1000/minute for Professional, and custom limits for Enterprise. Rate limit headers are included in responses. Exceeded requests return HTTP 429.",
        metadata={"category": "technical", "difficulty": "hard"},
    ),
    EvaluationSample(
        question="What payment methods are accepted?",
        ground_truth="The platform accepts credit cards (Visa, MasterCard, American Express), PayPal, bank transfers (for Enterprise), and cryptocurrency (Bitcoin, Ethereum) for annual subscriptions.",
        metadata={"category": "billing", "difficulty": "easy"},
    ),
    EvaluationSample(
        question="How can I export my data from the platform?",
        ground_truth="Users can export data through the Settings > Data Export page. Available formats include JSON, CSV, and PDF. Exports are processed asynchronously and emailed when ready. Large exports may take several hours.",
        metadata={"category": "features", "difficulty": "medium"},
    ),
]


# ============================================================
# RAG Evaluator
# ============================================================

class RAGEvaluator:
    """
    RAG system evaluator using RAGAS metrics.

    Evaluates RAG systems on four key dimensions:
    - Faithfulness: Factual consistency of answer with retrieved context
    - Answer Relevancy: Relevance of answer to the question
    - Context Precision: Proportion of relevant retrieved contexts
    - Context Recall: Proportion of ground truth covered by contexts

    Args:
        rag_chain: RAGChain instance to evaluate
        enable_evaluation: Override settings.ENABLE_EVALUATION

    Example:
        >>> evaluator = RAGEvaluator(rag_chain)
        >>> result = evaluator.evaluate_samples(DEFAULT_TEST_SAMPLES)
        >>> report = evaluator.generate_report(result)
    """

    def __init__(self, rag_chain: Any, enable_evaluation: Optional[bool] = None) -> None:
        """
        Initialize the RAG evaluator.

        Args:
            rag_chain: RAGChain instance to evaluate
            enable_evaluation: Override settings evaluation flag
        """
        self.rag_chain = rag_chain
        self.enable_evaluation = enable_evaluation if enable_evaluation is not None else settings.ENABLE_EVALUATION

        if not self.enable_evaluation:
            logger.warning("RAGAS evaluation is disabled. Set ENABLE_EVALUATION=True to enable.")

        self._ragas_metrics: Optional[dict[str, Any]] = None

        logger.info(
            "RAGEvaluator initialized",
            extra={"enabled": self.enable_evaluation},
        )

    @property
    def ragas_metrics(self) -> dict[str, Any]:
        """
        Lazy load RAGAS metrics.

        Returns:
            Dictionary of metric name to metric object

        Example:
            >>> metrics = evaluator.ragas_metrics
            >>> print(list(metrics.keys()))
        """
        if not self.enable_evaluation:
            raise RuntimeError("RAGAS evaluation is disabled")

        if self._ragas_metrics is None:
            try:
                from ragas.metrics import (
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    faithfulness,
                )

                self._ragas_metrics = {
                    "faithfulness": faithfulness,
                    "answer_relevancy": answer_relevancy,
                    "context_precision": context_precision.with_context(
                        context_recall
                    ),  # Use with_recall variant
                    "context_recall": context_recall,
                }

                logger.info("RAGAS metrics loaded successfully")

            except ImportError as e:
                logger.error(f"Failed to import RAGAS: {str(e)}")
                raise ImportError(
                    "RAGAS package is required for evaluation. "
                    "Install with: pip install ragas"
                )
            except Exception as e:
                logger.error(f"Failed to load RAGAS metrics: {str(e)}", exc_info=True)
                raise

        return self._ragas_metrics

    # ============================================================
    # Dataset Creation
    # ============================================================

    def create_test_dataset(
        self,
        samples: list[EvaluationSample],
        batch_size: int = 10,
        show_progress: bool = True,
    ) -> "Dataset":
        """
        Create a RAGAS-compatible dataset from evaluation samples.

        For samples without answers, generates them using the RAG chain.

        Args:
            samples: List of EvaluationSample objects
            batch_size: Batch size for answer generation
            show_progress: Whether to show progress bar

        Returns:
            RAGAS Dataset object

        Example:
            >>> dataset = evaluator.create_test_dataset(samples)
            >>> result = evaluator.evaluate_dataset(dataset)
        """
        try:
            from datasets import Dataset
            from tqdm import tqdm
        except ImportError:
            logger.error("datasets and tqdm packages are required")
            raise

        # Prepare data
        data = {
            "question": [],
            "ground_truth": [],
            "contexts": [],
            "answer": [],
        }

        # Generate answers for samples that don't have them
        samples_to_process = [s for s in samples if s.answer is None]
        samples_with_answers = [s for s in samples if s.answer is not None]

        if samples_to_process:
            logger.info(f"Generating answers for {len(samples_to_process)} samples")

            iterator = tqdm(samples_to_process, desc="Generating answers") if show_progress else samples_to_process

            for sample in iterator:
                try:
                    # Query the RAG chain
                    response = self.rag_chain.query(sample.question)

                    # Update sample
                    sample.answer = response.answer
                    sample.contexts = [r.content for r in response.retrieval_results]

                    # Add to data
                    data["question"].append(sample.question)
                    data["ground_truth"].append([sample.ground_truth])  # RAGAS expects list
                    data["contexts"].append(sample.contexts)
                    data["answer"].append(sample.answer)

                except Exception as e:
                    logger.error(f"Failed to generate answer for question: {sample.question[:50]}...", exc_info=True)
                    # Still add but with empty answer
                    data["question"].append(sample.question)
                    data["ground_truth"].append([sample.ground_truth])
                    data["contexts"].append([])
                    data["answer"].append("")

        # Add samples that already have answers
        for sample in samples_with_answers:
            data["question"].append(sample.question)
            data["ground_truth"].append([sample.ground_truth])
            data["contexts"].append(sample.contexts or [])
            data["answer"].append(sample.answer)

        logger.info(f"Created dataset with {len(data['question'])} samples")

        return Dataset.from_dict(data)

    # ============================================================
    # Evaluation Methods
    # ============================================================

    def evaluate_samples(
        self,
        samples: list[EvaluationSample],
        metrics: Optional[list[str]] = None,
        show_progress: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate RAG system on a list of samples.

        Args:
            samples: List of EvaluationSample objects
            metrics: List of metric names (default: all)
            show_progress: Whether to show progress

        Returns:
            EvaluationResult with all scores

        Example:
            >>> result = evaluator.evaluate_samples(samples)
            >>> print(f"Overall: {result.overall_score:.3f}")
        """
        if not self.enable_evaluation:
            logger.warning("Evaluation is disabled, returning dummy result")
            return EvaluationResult(
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0,
                overall_score=0.0,
                individual_results=[],
                evaluation_time=0.0,
                num_samples=len(samples),
            )

        # Create dataset
        dataset = self.create_test_dataset(samples, show_progress=show_progress)

        # Run evaluation
        return self.evaluate_dataset(dataset, metrics=metrics, show_progress=show_progress)

    def evaluate_dataset(
        self,
        dataset: "Dataset",
        metrics: Optional[list[str]] = None,
        show_progress: bool = True,
    ) -> EvaluationResult:
        """
        Run RAGAS evaluation on a dataset.

        Args:
            dataset: RAGAS Dataset with questions, answers, contexts, ground_truth
            metrics: List of metric names to compute (default: all)
            show_progress: Whether to show progress

        Returns:
            EvaluationResult with all metric scores

        Example:
            >>> result = evaluator.evaluate_dataset(test_dataset)
            >>> print(result.faithfulness)
        """
        if not self.enable_evaluation:
            raise RuntimeError("RAGAS evaluation is disabled")

        start_time = time.time()
        logger.info(f"Starting evaluation on {len(dataset)} samples")

        try:
            from ragas import evaluate

            # Determine which metrics to use
            available_metrics = self.ragas_metrics
            metrics_to_use = metrics or list(available_metrics.keys())

            # Validate metrics
            for m in metrics_to_use:
                if m not in available_metrics:
                    raise ValueError(f"Unknown metric: {m}. Available: {list(available_metrics.keys())}")

            # Get metric objects
            metric_objects = [available_metrics[m] for m in metrics_to_use]

            # Run evaluation
            results = evaluate(
                dataset,
                metrics=metric_objects,
                show_progress=show_progress,
            )

            # Calculate overall score
            scores_dict = results.to_dict()
            metric_scores = [scores_dict.get(m, 0.0) for m in metrics_to_use]
            overall_score = sum(metric_scores) / len(metric_scores) if metric_scores else 0.0

            # Get individual results
            individual_results = results.to_pandas().to_dict("records")

            evaluation_time = time.time() - start_time

            result = EvaluationResult(
                faithfulness=scores_dict.get("faithfulness", 0.0),
                answer_relevancy=scores_dict.get("answer_relevancy", 0.0),
                context_precision=scores_dict.get("context_precision", 0.0),
                context_recall=scores_dict.get("context_recall", 0.0),
                overall_score=overall_score,
                individual_results=individual_results,
                evaluation_time=evaluation_time,
                num_samples=len(dataset),
            )

            logger.info(
                f"Evaluation complete: {result.overall_score:.3f} overall score",
                extra={
                    "faithfulness": result.faithfulness,
                    "answer_relevancy": result.answer_relevancy,
                    "context_precision": result.context_precision,
                    "context_recall": result.context_recall,
                    "time": round(evaluation_time, 2),
                },
            )

            return result

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
            raise

    def evaluate_single(
        self,
        question: str,
        ground_truth: str,
    ) -> dict[str, Any]:
        """
        Evaluate a single question-answer pair.

        Args:
            question: Question to evaluate
            ground_truth: Reference answer

        Returns:
            Dictionary with metric scores

        Example:
            >>> scores = evaluator.evaluate_single(
            ...     "What is the refund policy?",
            ...     "Refunds take 5-7 business days."
            ... )
            >>> print(scores["faithfulness"])
        """
        sample = EvaluationSample(question=question, ground_truth=ground_truth)
        result = self.evaluate_samples([sample], show_progress=False)

        if result.individual_results:
            return result.individual_results[0]
        else:
            return {}

    # ============================================================
    # Reporting
    # ============================================================

    def generate_report(self, result: EvaluationResult) -> str:
        """
        Generate a markdown evaluation report.

        Args:
            result: EvaluationResult to report on

        Returns:
            Markdown formatted report

        Example:
            >>> report = evaluator.generate_report(result)
            >>> print(report)
            >>> # Save to file
            >>> Path("report.md").write_text(report)
        """
        report = f"""# RAG Evaluation Report

Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Overall Scores

| Metric | Score | Rating |
|--------|-------|--------|
| **Faithfulness** | {result.faithfulness:.3f} | {self._get_rating(result.faithfulness)} |
| **Answer Relevancy** | {result.answer_relevancy:.3f} | {self._get_rating(result.answer_relevancy)} |
| **Context Precision** | {result.context_precision:.3f} | {self._get_rating(result.context_precision)} |
| **Context Recall** | {result.context_recall:.3f} | {self._get_rating(result.context_recall)} |
| **Overall** | **{result.overall_score:.3f}** | **{self._get_rating(result.overall_score)}** |

## Evaluation Details

- **Samples evaluated:** {result.num_samples}
- **Evaluation time:** {result.evaluation_time:.2f} seconds
- **Avg time per sample:** {result.evaluation_time / result.num_samples:.2f} seconds

## Metric Interpretation

### Faithfulness
Measures the factual consistency of the generated answer with the retrieved context.
- **Score:** How well the answer stays true to the provided context
- **High score:** Answer is factually consistent with context
- **Low score:** Answer contains hallucinations or contradictions

### Answer Relevancy
Measures how relevant the answer is to the question.
- **Score:** How well the answer addresses the question
- **High score:** Answer directly addresses the question
- **Low score:** Answer is vague, incomplete, or off-topic

### Context Precision
Measures the signal-to-noise ratio in the retrieved contexts.
- **Score:** Proportion of retrieved contexts that are relevant
- **High score:** Most retrieved contexts are relevant
- **Low score:** Many irrelevant contexts were retrieved

### Context Recall
Measures how well the retrieved contexts cover the ground truth.
- **Score:** Proportion of ground truth information present in contexts
- **High score:** All necessary information was retrieved
- **Low score:** Important information was missing from contexts

## Recommendations

"""

        # Add recommendations based on scores
        recommendations = self._generate_recommendations(result)
        for rec in recommendations:
            report += f"- {rec}\n"

        report += "\n## Individual Results\n\n"

        # Add individual results
        for i, individual in enumerate(result.individual_results, 1):
            report += f"### Sample {i}\n\n"
            for key, value in individual.items():
                if isinstance(value, float):
                    report += f"- **{key}**: {value:.3f}\n"
                elif isinstance(value, str) and len(value) > 100:
                    report += f"- **{key}**: {value[:100]}...\n"
                else:
                    report += f"- **{key}**: {value}\n"
            report += "\n"

        return report

    def _get_rating(self, score: float) -> str:
        """Get rating label for score."""
        if score >= 0.9:
            return "Excellent ⭐⭐⭐⭐⭐"
        elif score >= 0.75:
            return "Good ⭐⭐⭐⭐"
        elif score >= 0.6:
            return "Fair ⭐⭐⭐"
        elif score >= 0.4:
            return "Poor ⭐⭐"
        else:
            return "Very Poor ⭐"

    def _generate_recommendations(self, result: EvaluationResult) -> list[str]:
        """Generate recommendations based on scores."""
        recommendations = []

        if result.faithfulness < 0.7:
            recommendations.append(
                "⚠️ **Low Faithfulness**: Consider improving prompt engineering to reduce hallucinations. "
                "Add explicit instructions to only use provided context."
            )

        if result.answer_relevancy < 0.7:
            recommendations.append(
                "⚠️ **Low Answer Relevancy**: The model may not be addressing questions directly. "
                "Consider refining the system prompt or using a more capable model."
            )

        if result.context_precision < 0.7:
            recommendations.append(
                "⚠️ **Low Context Precision**: Retrieved contexts contain irrelevant information. "
                "Consider adjusting retrieval parameters (top_k, reranking) or improving document chunking."
            )

        if result.context_recall < 0.7:
            recommendations.append(
                "⚠️ **Low Context Recall**: Important information is being missed. "
                "Consider increasing top_k_retrieve or improving the document index."
            )

        if result.overall_score >= 0.8:
            recommendations.append(
                "✅ **Good Performance**: Your RAG system is performing well! "
                "Continue monitoring and evaluating on diverse question types."
            )

        if not recommendations:
            recommendations.append("✅ **All metrics are acceptable** - Continue monitoring.")

        return recommendations

    # ============================================================
    # Dataset Management
    # ============================================================

    def load_test_dataset(self, path: str) -> list[EvaluationSample]:
        """
        Load test dataset from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            List of EvaluationSample objects

        Example:
            >>> samples = evaluator.load_test_dataset("test_data.json")
            >>> result = evaluator.evaluate_samples(samples)
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Test dataset not found: {path}")

        logger.info(f"Loading test dataset from {path}")

        with open(file_path, "r") as f:
            data = json.load(f)

        samples = [EvaluationSample(**item) for item in data]

        logger.info(f"Loaded {len(samples)} samples")

        return samples

    def save_test_dataset(self, samples: list[EvaluationSample], path: str) -> None:
        """
        Save test dataset to JSON file.

        Args:
            samples: List of EvaluationSample objects
            path: Path to save

        Example:
            >>> evaluator.save_test_dataset(samples, "test_data.json")
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        data = [sample.to_dict() for sample in samples]

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(samples)} samples to {path}")

    def save_results(self, result: EvaluationResult, path: str) -> None:
        """
        Save evaluation results to JSON file.

        Args:
            result: EvaluationResult to save
            path: Path to save

        Example:
            >>> evaluator.save_results(result, "evaluation_results.json")
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Saved evaluation results to {path}")

    def save_report(self, result: EvaluationResult, path: str) -> None:
        """
        Save evaluation report to markdown file.

        Args:
            result: EvaluationResult to report on
            path: Path to save report

        Example:
            >>> evaluator.save_report(result, "report.md")
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        report = self.generate_report(result)

        with open(file_path, "w") as f:
            f.write(report)

        logger.info(f"Saved evaluation report to {path}")

    # ============================================================
    # Default Dataset
    # ============================================================

    @classmethod
    def get_default_samples(cls) -> list[EvaluationSample]:
        """
        Get the default test dataset.

        Returns:
            List of 10 default evaluation samples

        Example:
            >>> samples = RAGEvaluator.get_default_samples()
            >>> print(len(samples))  # 10
        """
        return DEFAULT_TEST_SAMPLES.copy()


# ============================================================
# Utility Functions
# ============================================================

def create_evaluator(rag_chain: Any) -> RAGEvaluator:
    """
    Create a RAG evaluator configured from settings.

    Args:
        rag_chain: RAGChain instance

    Returns:
        Configured RAGEvaluator instance

    Example:
        >>> evaluator = create_evaluator(rag_chain)
        >>> result = evaluator.evaluate_samples(samples)
    """
    return RAGEvaluator(rag_chain=rag_chain)


def quick_evaluation(
    rag_chain: Any,
    num_samples: int = 10,
) -> EvaluationResult:
    """
    Run a quick evaluation on default samples.

    Args:
        rag_chain: RAGChain to evaluate
        num_samples: Number of samples to evaluate (max 10)

    Returns:
        EvaluationResult with scores

    Example:
        >>> result = quick_evaluation(rag_chain, num_samples=5)
        >>> print(f"Overall: {result.overall_score:.3f}")
    """
    evaluator = create_evaluator(rag_chain)
    samples = RAGEvaluator.get_default_samples()[:num_samples]
    return evaluator.evaluate_samples(samples)


# Export public API
__all__ = [
    # Classes
    "RAGEvaluator",
    "EvaluationSample",
    "EvaluationResult",
    # Constants
    "DEFAULT_TEST_SAMPLES",
    # Utilities
    "create_evaluator",
    "quick_evaluation",
]
