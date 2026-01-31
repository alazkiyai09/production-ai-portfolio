"""
Experiment orchestration framework for prompt optimization.

This module provides the main experiment framework that coordinates
prompt variation generation, evaluation execution, and statistical analysis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
import asyncio
from datetime import datetime
from pathlib import Path
import json
import logging
import uuid

from src.prompt_optimizer.variations.variation_generator import (
    VariationGenerator,
    VariationSet,
    PromptVariation,
    VariationStrategy,
)
from src.prompt_optimizer.experiments.ab_test import (
    ABTester,
    ABTestConfig,
    TestResult,
    TestMethod,
    MultipleComparisonCorrection,
)
from src.runners.eval_runner import EvaluationRunner, EvaluationConfig, EvaluationResult
from src.datasets.dataset_manager import DatasetManager, TestDataset
from src.models.llm_providers import create_provider

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Classes
# ============================================================================

class ExperimentStatus(Enum):
    """Status of an experiment."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentConfig:
    """
    Configuration for a prompt optimization experiment.

    Attributes:
        name: Experiment name
        description: Human-readable description
        base_template_id: ID of base template to optimize
        dataset_name: Name of dataset to use for testing
        strategies: List of variation strategies to test
        variations_per_strategy: Number of variations per strategy
        combine_strategies: Whether to create combined variations
        sample_size: Sample size from dataset (None = use all)
        metrics: List of metrics to evaluate
        provider: LLM provider name
        model: Model name
        test_config: A/B test configuration
        max_concurrency: Maximum concurrent evaluations
        output_dir: Output directory for results
    """

    name: str
    description: str
    base_template_id: str
    dataset_name: str
    strategies: List[VariationStrategy]
    variations_per_strategy: int = 3
    combine_strategies: bool = False
    sample_size: Optional[int] = None
    metrics: List[str] = field(default_factory=lambda: ["semantic_similarity", "llm_judge"])
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    test_config: ABTestConfig = field(default_factory=ABTestConfig)
    max_concurrency: int = 10
    output_dir: str = "data/experiments"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "base_template_id": self.base_template_id,
            "dataset_name": self.dataset_name,
            "strategies": [s.value for s in self.strategies],
            "variations_per_strategy": self.variations_per_strategy,
            "combine_strategies": self.combine_strategies,
            "sample_size": self.sample_size,
            "metrics": self.metrics,
            "provider": self.provider,
            "model": self.model,
            "test_config": {
                "alpha": self.test_config.alpha,
                "power": self.test_config.power,
                "effect_size_threshold": self.test_config.effect_size_threshold,
                "correction_method": self.test_config.correction_method.value,
                "min_sample_size": self.test_config.min_sample_size,
                "test_method": self.test_config.test_method.value,
            },
            "max_concurrency": self.max_concurrency,
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        # Convert test config
        test_config_data = data.get("test_config", {})
        test_config = ABTestConfig(
            alpha=test_config_data.get("alpha", 0.05),
            power=test_config_data.get("power", 0.80),
            effect_size_threshold=test_config_data.get("effect_size_threshold", 0.5),
            correction_method=MultipleComparisonCorrection(
                test_config_data.get("correction_method", "bonferroni")
            ),
            min_sample_size=test_config_data.get("min_sample_size", 30),
            test_method=TestMethod(test_config_data.get("test_method", "t_test")),
        )

        # Convert strategies
        strategies = [
            VariationStrategy(s) for s in data.get("strategies", [])
        ]

        return cls(
            name=data["name"],
            description=data["description"],
            base_template_id=data["base_template_id"],
            dataset_name=data["dataset_name"],
            strategies=strategies,
            variations_per_strategy=data.get("variations_per_strategy", 3),
            combine_strategies=data.get("combine_strategies", False),
            sample_size=data.get("sample_size"),
            metrics=data.get("metrics", ["semantic_similarity", "llm_judge"]),
            provider=data.get("provider", "openai"),
            model=data.get("model", "gpt-4o-mini"),
            test_config=test_config,
            max_concurrency=data.get("max_concurrency", 10),
            output_dir=data.get("output_dir", "data/experiments"),
        )


@dataclass
class VariantResult:
    """
    Results for a single variant evaluation.

    Attributes:
        variant_id: Variant identifier
        variant_name: Human-readable name
        strategy: Strategy used
        prompt_content: The prompt that was tested
        system_prompt: System prompt if used
        scores: Dict of metric name to score list
        mean_scores: Dict of metric name to mean score
        std_scores: Dict of metric name to std score
        evaluation_time: Time taken for evaluation
        token_usage: Total tokens used
        cost: Total cost
        metadata: Additional metadata
    """

    variant_id: str
    variant_name: str
    strategy: VariationStrategy
    prompt_content: str
    system_prompt: Optional[str]
    scores: Dict[str, List[float]]
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    evaluation_time: float
    token_usage: Dict[str, int]
    cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def avg_score(self) -> float:
        """Get average score across all metrics."""
        if not self.mean_scores:
            return 0.0
        return sum(self.mean_scores.values()) / len(self.mean_scores)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variant_id": self.variant_id,
            "variant_name": self.variant_name,
            "strategy": self.strategy.value,
            "prompt_content": self.prompt_content,
            "system_prompt": self.system_prompt,
            "scores": self.scores,
            "mean_scores": self.mean_scores,
            "std_scores": self.std_scores,
            "avg_score": self.avg_score,
            "evaluation_time": self.evaluation_time,
            "token_usage": self.token_usage,
            "cost": self.cost,
            "metadata": self.metadata,
        }


@dataclass
class ExperimentResult:
    """
    Complete results of an experiment.

    Attributes:
        experiment_id: Unique experiment identifier
        config: Experiment configuration
        status: Experiment status
        variation_set: Set of prompts tested
        variant_results: Results for each variant
        statistical_tests: Statistical test results by metric
        best_variant: ID of best performing variant
        rankings: Variant rankings by metric
        started_at: Start timestamp
        completed_at: Completion timestamp
        total_time: Total experiment time
        metadata: Additional metadata
    """

    experiment_id: str
    config: ExperimentConfig
    status: ExperimentStatus
    variation_set: VariationSet
    variant_results: Dict[str, VariantResult]
    statistical_tests: Dict[str, Dict[str, TestResult]]  # metric -> variant -> result
    best_variant: Optional[str]
    rankings: Dict[str, List[Tuple[str, float]]]  # metric -> [(variant_id, score)]
    started_at: str
    completed_at: Optional[str]
    total_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "variation_set": self.variation_set.to_dict(),
            "variant_results": {
                k: v.to_dict() for k, v in self.variant_results.items()
            },
            "statistical_tests": {
                metric: {
                    variant: result.to_dict()
                    for variant, result in variants.items()
                }
                for metric, variants in self.statistical_tests.items()
            },
            "best_variant": self.best_variant,
            "rankings": self.rankings,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_time": self.total_time,
            "metadata": self.metadata,
        }

    def save(self, path: Optional[str] = None) -> str:
        """
        Save experiment results to file.

        Args:
            path: Output path (defaults to config output_dir)

        Returns:
            Path to saved file
        """
        if path is None:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{self.experiment_id}.json"

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        return str(path)

    @classmethod
    def load(cls, path: str) -> "ExperimentResult":
        """Load experiment results from file."""
        with open(path, "r") as f:
            data = json.load(f)

        config = ExperimentConfig.from_dict(data["config"])
        variation_set = VariationSet.from_dict(data["variation_set"])

        # Reconstruct variant results
        variant_results = {}
        for var_id, var_data in data["variant_results"].items():
            variant_results[var_id] = VariantResult(
                variant_id=var_data["variant_id"],
                variant_name=var_data["variant_name"],
                strategy=VariationStrategy(var_data["strategy"]),
                prompt_content=var_data["prompt_content"],
                system_prompt=var_data.get("system_prompt"),
                scores=var_data["scores"],
                mean_scores=var_data["mean_scores"],
                std_scores=var_data["std_scores"],
                evaluation_time=var_data["evaluation_time"],
                token_usage=var_data["token_usage"],
                cost=var_data["cost"],
                metadata=var_data.get("metadata", {}),
            )

        return cls(
            experiment_id=data["experiment_id"],
            config=config,
            status=ExperimentStatus(data["status"]),
            variation_set=variation_set,
            variant_results=variant_results,
            statistical_tests=data.get("statistical_tests", {}),
            best_variant=data.get("best_variant"),
            rankings=data.get("rankings", {}),
            started_at=data["started_at"],
            completed_at=data.get("completed_at"),
            total_time=data["total_time"],
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# Main Experiment Framework
# ============================================================================

class ExperimentFramework:
    """
    Main experiment orchestration framework.

    Coordinates:
    1. Prompt variation generation
    2. Evaluation execution
    3. Statistical analysis
    4. Result aggregation
    """

    def __init__(
        self,
        variation_generator: VariationGenerator,
        dataset_manager: DatasetManager,
        evaluation_runner: EvaluationRunner,
        ab_tester: ABTester,
    ):
        """
        Initialize the experiment framework.

        Args:
            variation_generator: For generating prompt variations
            dataset_manager: For loading test datasets
            evaluation_runner: For running evaluations
            ab_tester: For statistical analysis
        """
        self.variation_generator = variation_generator
        self.dataset_manager = dataset_manager
        self.evaluation_runner = evaluation_runner
        self.ab_tester = ab_tester

    async def run_experiment(
        self,
        config: ExperimentConfig,
        base_prompt: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> ExperimentResult:
        """
        Run a complete experiment.

        Args:
            config: Experiment configuration
            base_prompt: Base prompt to optimize
            progress_callback: Optional callback for progress updates

        Returns:
            Complete experiment results
        """
        experiment_id = str(uuid.uuid4())[:8]
        started_at = datetime.utcnow().isoformat()
        logger.info(f"Starting experiment {experiment_id}: {config.name}")

        try:
            # Update progress
            if progress_callback:
                progress_callback("Loading dataset", 0.1)

            # Load dataset
            dataset = self.dataset_manager.load_dataset(config.dataset_name)
            if config.sample_size:
                dataset = dataset.sample(n=config.sample_size, seed=42)

            logger.info(f"Loaded dataset: {len(dataset.test_cases)} test cases")

            # Generate variations
            if progress_callback:
                progress_callback("Generating prompt variations", 0.2)

            variation_set = self.variation_generator.generate(
                base_prompt=base_prompt,
                strategies=config.strategies,
                variations_per_strategy=config.variations_per_strategy,
                combine_strategies=config.combine_strategies,
            )

            logger.info(f"Generated {len(variation_set.variations)} prompt variations")

            # Evaluate each variant
            if progress_callback:
                progress_callback("Evaluating variants", 0.3)

            variant_results = {}
            total_variants = len(variation_set.variations)

            for i, variation in enumerate(variation_set.variations):
                progress = 0.3 + (i / total_variants) * 0.5
                if progress_callback:
                    progress_callback(
                        f"Evaluating variant {i+1}/{total_variants}: {variation.description}",
                        progress,
                    )

                logger.info(f"Evaluating variant: {variation.id}")

                # Evaluate this variant
                result = await self._evaluate_variant(
                    variation=variation,
                    dataset=dataset,
                    config=config,
                )

                variant_results[variation.id] = result

            # Statistical analysis
            if progress_callback:
                progress_callback("Running statistical analysis", 0.85)

            statistical_tests = await self._run_statistical_tests(
                variant_results=variant_results,
                control_id="base",  # First variant is control
                config=config,
            )

            # Determine rankings and best variant
            if progress_callback:
                progress_callback("Calculating rankings", 0.95)

            rankings, best_variant = self._calculate_rankings(
                variant_results=variant_results,
                statistical_tests=statistical_tests,
                config=config,
            )

            completed_at = datetime.utcnow().isoformat()
            total_time = (
                datetime.fromisoformat(completed_at) -
                datetime.fromisoformat(started_at)
            ).total_seconds()

            # Create result
            result = ExperimentResult(
                experiment_id=experiment_id,
                config=config,
                status=ExperimentStatus.COMPLETED,
                variation_set=variation_set,
                variant_results=variant_results,
                statistical_tests=statistical_tests,
                best_variant=best_variant,
                rankings=rankings,
                started_at=started_at,
                completed_at=completed_at,
                total_time=total_time,
                metadata={
                    "total_variants": total_variants,
                    "total_test_cases": len(dataset.test_cases),
                },
            )

            # Save results
            result_path = result.save()
            logger.info(f"Experiment results saved to: {result_path}")

            if progress_callback:
                progress_callback("Experiment complete", 1.0)

            return result

        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            completed_at = datetime.utcnow().isoformat()

            return ExperimentResult(
                experiment_id=experiment_id,
                config=config,
                status=ExperimentStatus.FAILED,
                variation_set=VariationSet(
                    id="failed",
                    name="Failed",
                    base_template_id=config.base_template_id,
                ),
                variant_results={},
                statistical_tests={},
                best_variant=None,
                rankings={},
                started_at=started_at,
                completed_at=completed_at,
                total_time=0.0,
                metadata={"error": str(e)},
            )

    async def _evaluate_variant(
        self,
        variation: PromptVariation,
        dataset: TestDataset,
        config: ExperimentConfig,
    ) -> VariantResult:
        """Evaluate a single prompt variant."""
        import time

        start_time = time.time()

        # Create evaluation config
        eval_config = EvaluationConfig(
            dataset_name=config.dataset_name,
            metrics=config.metrics,
            provider=config.provider,
            model=config.model,
            max_concurrency=config.max_concurrency,
        )

        # Create custom prompt function
        async def prompt_fn(test_case):
            # Render prompt with test case
            # For simplicity, just use the variant's prompt content
            return variation.prompt_content

        # Run evaluation
        eval_result = await self.evaluation_runner.evaluate_dataset(
            dataset=dataset,
            config=eval_config,
            prompt_fn=prompt_fn,
            system_prompt=variation.system_prompt,
        )

        elapsed = time.time() - start_time

        # Extract scores by metric
        scores: Dict[str, List[float]] = {}
        for metric_name in config.metrics:
            scores[metric_name] = [
                result.metrics.get(metric_name, 0.0)
                for result in eval_result.test_results
            ]

        # Calculate means and stds
        mean_scores = {
            metric: (np.mean(vals) if vals else 0.0)
            for metric, vals in scores.items()
        }
        std_scores = {
            metric: (np.std(vals) if vals else 0.0)
            for metric, vals in scores.items()
        }

        # Extract token usage and cost
        token_usage = {
            "prompt_tokens": sum(
                r.metadata.get("prompt_tokens", 0) for r in eval_result.test_results
            ),
            "completion_tokens": sum(
                r.metadata.get("completion_tokens", 0) for r in eval_result.test_results
            ),
        }
        token_usage["total_tokens"] = (
            token_usage["prompt_tokens"] + token_usage["completion_tokens"]
        )

        cost = sum(r.metadata.get("cost", 0.0) for r in eval_result.test_results)

        return VariantResult(
            variant_id=variation.id,
            variant_name=variation.description or variation.id,
            strategy=variation.strategy,
            prompt_content=variation.prompt_content,
            system_prompt=variation.system_prompt,
            scores=scores,
            mean_scores=mean_scores,
            std_scores=std_scores,
            evaluation_time=elapsed,
            token_usage=token_usage,
            cost=cost,
            metadata={
                "num_test_cases": len(eval_result.test_results),
                "variation_params": variation.variation_params,
            },
        )

    async def _run_statistical_tests(
        self,
        variant_results: Dict[str, VariantResult],
        control_id: str,
        config: ExperimentConfig,
    ) -> Dict[str, Dict[str, TestResult]]:
        """Run statistical tests comparing variants to control."""
        tests_by_metric: Dict[str, Dict[str, TestResult]] = {}

        # Get control scores
        if control_id not in variant_results:
            logger.warning(f"Control variant '{control_id}' not found, using first variant")
            control_id = list(variant_results.keys())[0]

        control_result = variant_results[control_id]

        # Test each metric
        for metric in config.metrics:
            tests_by_metric[metric] = {}

            # Compare each variant to control
            for variant_id, variant_result in variant_results.items():
                if variant_id == control_id:
                    continue

                control_scores = control_result.scores.get(metric, [])
                treatment_scores = variant_result.scores.get(metric, [])

                if not control_scores or not treatment_scores:
                    continue

                # Run test
                test_result = self.ab_tester.compare_variants(
                    control_scores=control_scores,
                    treatment_scores=treatment_scores,
                    variant_name=variant_id,
                )

                tests_by_metric[metric][variant_id] = test_result

        return tests_by_metric

    def _calculate_rankings(
        self,
        variant_results: Dict[str, VariantResult],
        statistical_tests: Dict[str, Dict[str, TestResult]],
        config: ExperimentConfig,
    ) -> Tuple[Dict[str, List[Tuple[str, float]]], Optional[str]]:
        """Calculate variant rankings and determine best variant."""
        rankings: Dict[str, List[Tuple[str, float]]] = {}

        # Rank by each metric
        for metric in config.metrics:
            # Get mean scores for all variants
            scores = [
                (var_id, result.mean_scores.get(metric, 0.0))
                for var_id, result in variant_results.items()
            ]

            # Sort by score (descending)
            scores.sort(key=lambda x: x[1], reverse=True)
            rankings[metric] = scores

        # Determine best variant (highest average score across metrics)
        if rankings:
            # Calculate average rank across all metrics
            variant_ranks: Dict[str, List[int]] = {}

            for metric_ranking in rankings.values():
                for rank, (var_id, _) in enumerate(metric_ranking):
                    if var_id not in variant_ranks:
                        variant_ranks[var_id] = []
                    variant_ranks[var_id].append(rank)

            # Get variant with best average rank
            avg_ranks = {
                var_id: sum(ranks) / len(ranks)
                for var_id, ranks in variant_ranks.items()
            }

            best_variant = min(avg_ranks, key=avg_ranks.get)
        else:
            best_variant = None

        return rankings, best_variant


# ============================================================================
# Convenience Functions
# ============================================================================

def create_experiment_framework(
    variation_seed: int = 42,
    test_alpha: float = 0.05,
    test_power: float = 0.80,
    test_effect_size: float = 0.5,
    correction_method: MultipleComparisonCorrection = MultipleComparisonCorrection.BONFERRONI,
) -> ExperimentFramework:
    """
    Factory function to create an experiment framework.

    Args:
        variation_seed: Seed for variation generation
        test_alpha: Significance level for tests
        test_power: Statistical power
        test_effect_size: Effect size threshold
        correction_method: Multiple comparison correction method

    Returns:
        Configured ExperimentFramework
    """
    # Create components
    from src.prompt_optimizer.variations.variation_generator import create_variation_generator
    from src.datasets.dataset_manager import create_dataset_manager
    from src.runners.eval_runner import create_evaluation_runner
    from src.prompt_optimizer.experiments.ab_test import create_ab_tester

    variation_gen = create_variation_generator(seed=variation_seed)
    dataset_mgr = create_dataset_manager()
    eval_runner = create_evaluation_runner()
    ab_tester = create_ab_tester(
        alpha=test_alpha,
        power=test_power,
        effect_size_threshold=test_effect_size,
        correction_method=correction_method,
    )

    return ExperimentFramework(
        variation_generator=variation_gen,
        dataset_manager=dataset_mgr,
        evaluation_runner=eval_runner,
        ab_tester=ab_tester,
    )


# Import numpy for calculations
import numpy as np

# Export main classes and functions
__all__ = [
    # Enums
    "ExperimentStatus",
    # Data classes
    "ExperimentConfig",
    "VariantResult",
    "ExperimentResult",
    # Main class
    "ExperimentFramework",
    "create_experiment_framework",
]
