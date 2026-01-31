"""
Evaluation runner for executing LLM tests across models and datasets.

This module provides the core evaluation orchestration, including parallel
execution, progress tracking, results collection, and persistence.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Callable
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

# Import from our modules
from src.datasets.dataset_manager import (
    DatasetManager,
    TestDataset,
    TestCase,
)
from src.models.llm_providers import (
    create_provider,
    LLMProvider,
    LLMResponse,
)
from src.evaluation.metrics import (
    MetricResult,
    create_metric,
    evaluate_metrics,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class ModelConfig:
    """
    Configuration for a single model to evaluate.

    Attributes:
        provider: Provider name (openai, anthropic, ollama)
        model: Model identifier
        api_key: Optional API key (falls back to config)
        params: Additional generation parameters (temperature, etc.)
    """

    provider: str
    model: str
    api_key: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "params": self.params,
        }


@dataclass
class EvaluationConfig:
    """
    Configuration for an evaluation run.

    Attributes:
        name: Evaluation run name/identifier
        dataset: Dataset name to evaluate
        dataset_version: Dataset version (default: latest)
        models: List of model configurations to evaluate
        metrics: List of metric names to run
        categories: Optional category filter
        tags: Optional tag filter
        parallel: Maximum concurrent evaluations
        timeout_seconds: Timeout per test case
        save_results: Whether to save results to disk
        output_format: Output format (json, yaml, csv)
        include_failed: Include failed test cases in results
        max_retries: Retry failed evaluations
        sample_size: Optional sample size (None = all)
        random_seed: Random seed for sampling
    """

    name: str
    dataset: str
    models: list[ModelConfig]
    metrics: list[str]
    dataset_version: str = "latest"
    categories: Optional[list[str]] = None
    tags: Optional[list[str]] = None
    parallel: int = 5
    timeout_seconds: int = 60
    save_results: bool = True
    output_format: str = "json"
    include_failed: bool = True
    max_retries: int = 2
    sample_size: Optional[int] = None
    random_seed: int = 42
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "models": [m.to_dict() for m in self.models],
            "metrics": self.metrics,
            "categories": self.categories,
            "tags": self.tags,
            "parallel": self.parallel,
            "timeout_seconds": self.timeout_seconds,
            "save_results": self.save_results,
            "output_format": self.output_format,
            "include_failed": self.include_failed,
            "max_retries": self.max_retries,
            "sample_size": self.sample_size,
            "random_seed": self.random_seed,
            "metadata": self.metadata,
        }


@dataclass
class TestResult:
    """
    Result of a single test case evaluation.

    Attributes:
        test_id: Test case identifier
        model: Model name
        provider: Provider name
        prompt: Input prompt
        response: Generated response
        expected: Expected/reference response
        metrics: Dictionary of metric results
        latency_ms: Response latency in milliseconds
        cost_usd: Estimated cost in USD
        input_tokens: Input token count
        output_tokens: Output token count
        total_tokens: Total token count
        error: Error message if evaluation failed
        timestamp: Evaluation timestamp
        retry_count: Number of retries performed
        metadata: Additional metadata
    """

    test_id: str
    model: str
    provider: str
    prompt: str
    response: str
    expected: str
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Check if all metrics passed."""
        if self.error:
            return False
        return all(m.passed for m in self.metrics.values()) if self.metrics else True

    @property
    def overall_score(self) -> float:
        """Get overall score (average of metric values)."""
        if not self.metrics:
            return 0.0
        return sum(m.value for m in self.metrics.values()) / len(self.metrics)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_id": self.test_id,
            "model": self.model,
            "provider": self.provider,
            "prompt": self.prompt,
            "response": self.response,
            "expected": self.expected,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "error": self.error,
            "timestamp": self.timestamp,
            "retry_count": self.retry_count,
            "passed": self.passed,
            "overall_score": self.overall_score,
        }


@dataclass
class EvaluationResult:
    """
    Results of a complete evaluation run.

    Attributes:
        config: Evaluation configuration
        results: List of individual test results
        summary: Summary statistics
        start_time: Start timestamp
        end_time: End timestamp
        duration_seconds: Total duration
        total_cost_usd: Total cost across all evaluations
    """

    config: EvaluationConfig
    results: list[TestResult]
    summary: dict[str, Any]
    start_time: str
    end_time: str
    duration_seconds: float
    total_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": self.config.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "total_cost_usd": self.total_cost_usd,
        }


# ============================================================================
# Progress Tracking
# ============================================================================

class ProgressTracker:
    """
    Track and report evaluation progress.

    Can be extended with callbacks for real-time updates.
    """

    def __init__(
        self,
        total_tests: int,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        on_complete: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize progress tracker.

        Args:
            total_tests: Total number of tests to run
            on_progress: Callback(completed, total, current_model)
            on_complete: Callback when evaluation completes
        """
        self.total_tests = total_tests
        self.completed = 0
        self.failed = 0
        self.on_progress = on_progress
        self.on_complete = on_complete
        self._lock = asyncio.Lock()

    async def update(self, success: bool = True, model: str = "") -> None:
        """
        Update progress.

        Args:
            success: Whether the test succeeded
            model: Current model being evaluated
        """
        async with self._lock:
            self.completed += 1
            if not success:
                self.failed += 1

            if self.on_progress:
                await self.on_progress(self.completed, self.total_tests, model)

    async def complete(self) -> None:
        """Mark evaluation as complete."""
        if self.on_complete:
            await self.on_complete()

    @property
    def progress_percent(self) -> float:
        """Get progress percentage."""
        if self.total_tests == 0:
            return 100.0
        return (self.completed / self.total_tests) * 100

    @property
    def success_rate(self) -> float:
        """Get success rate."""
        if self.completed == 0:
            return 0.0
        return ((self.completed - self.failed) / self.completed) * 100


# ============================================================================
# Evaluation Runner
# ============================================================================

class EvaluationRunner:
    """
    Run evaluations across models and datasets.

    Handles parallel execution, progress tracking, error handling,
    and results persistence.
    """

    def __init__(
        self,
        output_dir: str | Path = "./data/results",
        datasets_dir: str | Path = "./data/datasets",
    ):
        """
        Initialize the evaluation runner.

        Args:
            output_dir: Directory to save results
            datasets_dir: Directory containing datasets
        """
        self.output_dir = Path(output_dir)
        self.datasets_dir = Path(datasets_dir)
        self.dataset_manager = DatasetManager(datasets_dir=self.datasets_dir)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, config: EvaluationConfig) -> EvaluationResult:
        """
        Run a complete evaluation.

        Args:
            config: Evaluation configuration

        Returns:
            EvaluationResult with all results and summary
        """
        start_time = datetime.utcnow()
        logger.info(f"Starting evaluation '{config.name}'")

        # Load and prepare dataset
        dataset = self._prepare_dataset(config)

        # Create providers
        providers = self._create_providers(config)

        # Calculate total tests
        total_tests = len(dataset.test_cases) * len(providers)

        # Setup progress tracking
        progress = ProgressTracker(
            total_tests=total_tests,
            on_progress=self._default_progress_callback,
        )

        # Run evaluations
        all_results = []
        for provider in providers:
            model_results = await self._evaluate_model(
                provider,
                dataset,
                config,
                progress,
            )
            all_results.extend(model_results)

        # Generate summary
        summary = self._generate_summary(all_results, config)

        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        # Calculate total cost
        total_cost = sum(r.cost_usd for r in all_results)

        # Create result
        result = EvaluationResult(
            config=config,
            results=all_results,
            summary=summary,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            total_cost_usd=total_cost,
        )

        # Save results
        if config.save_results:
            self._save_results(result)

        await progress.complete()
        logger.info(f"Evaluation '{config.name}' completed in {duration:.2f}s")

        return result

    def _prepare_dataset(self, config: EvaluationConfig) -> TestDataset:
        """Load and filter dataset based on config."""
        # Load dataset
        dataset = self.dataset_manager.load_dataset(
            config.dataset,
            version=config.dataset_version,
        )

        # Apply filters
        if config.categories:
            # Combine category filters
            filtered = None
            for category in config.categories:
                cat_ds = self.dataset_manager.filter_by_category(dataset, category)
                if filtered is None:
                    filtered = cat_ds
                else:
                    # Merge category results
                    filtered.test_cases.extend(cat_ds.test_cases)
            if filtered:
                dataset = filtered

        if config.tags:
            dataset = self.dataset_manager.filter_by_tags(
                dataset,
                config.tags,
                match_all=False,
            )

        # Sample if requested
        if config.sample_size and config.sample_size < len(dataset.test_cases):
            dataset = self.dataset_manager.sample(
                dataset,
                n=config.sample_size,
                random_seed=config.random_seed,
            )

        logger.info(f"Loaded dataset '{config.dataset}' with {len(dataset.test_cases)} test cases")

        return dataset

    def _create_providers(self, config: EvaluationConfig) -> list[LLMProvider]:
        """Create LLM provider instances."""
        providers = []
        for model_config in config.models:
            try:
                provider = create_provider(
                    provider=model_config.provider,
                    model=model_config.model,
                    api_key=model_config.api_key,
                )
                providers.append(provider)
                logger.info(f"Created {model_config.provider}:{model_config.model}")
            except Exception as e:
                logger.error(f"Failed to create provider {model_config.provider}:{model_config.model}: {e}")

        return providers

    async def _evaluate_model(
        self,
        provider: LLMProvider,
        dataset: TestDataset,
        config: EvaluationConfig,
        progress: ProgressTracker,
    ) -> list[TestResult]:
        """Evaluate a single model on the dataset."""
        semaphore = asyncio.Semaphore(config.parallel)

        async def evaluate_with_semaphore(test_case: TestCase) -> TestResult:
            async with semaphore:
                return await self._evaluate_test_case(
                    provider,
                    test_case,
                    config,
                )

        # Create tasks for all test cases
        tasks = [
            evaluate_with_semaphore(tc)
            for tc in dataset.test_cases
            if tc.enabled
        ]

        # Execute with progress tracking
        results = []
        for coro in asyncio.as_completed(tasks):
            try:
                result = await asyncio.wait_for(
                    coro,
                    timeout=config.timeout_seconds,
                )
                results.append(result)
                await progress.update(success=not bool(result.error), model=provider.model)
            except asyncio.TimeoutError:
                logger.warning(f"Test case timed out for {provider.model}")
                await progress.update(success=False, model=provider.model)
            except Exception as e:
                logger.error(f"Error evaluating test case: {e}")
                await progress.update(success=False, model=provider.model)

        return results

    async def _evaluate_test_case(
        self,
        provider: LLMProvider,
        test_case: TestCase,
        config: EvaluationConfig,
        retry_count: int = 0,
    ) -> TestResult:
        """
        Evaluate a single test case.

        Args:
            provider: LLM provider
            test_case: Test case to evaluate
            config: Evaluation config
            retry_count: Current retry attempt

        Returns:
            TestResult
        """
        try:
            # Generate response
            response: LLMResponse = await provider.generate(
                prompt=test_case.prompt,
                **test_case.context.get("generation_params", config.models[0].params),
            )

            # Determine which metrics to run
            metrics_to_run = test_case.get_effective_metrics(config.metrics)

            # Run metrics
            metric_context = {
                "latency_ms": response.latency_ms,
                "cost_usd": response.cost_usd,
                "generation_params": test_case.context.get("generation_params", {}),
                **test_case.context,
            }

            metric_results: dict[str, MetricResult] = {}
            for metric_name in metrics_to_run:
                try:
                    metric = create_metric(metric_name)
                    result = await metric.evaluate(
                        response=response.content,
                        expected=test_case.expected,
                        context=metric_context,
                    )
                    metric_results[metric_name] = result
                except Exception as e:
                    logger.warning(f"Metric {metric_name} failed: {e}")
                    metric_results[metric_name] = MetricResult(
                        name=metric_name,
                        value=0.0,
                        passed=False,
                        error=str(e),
                    )

            return TestResult(
                test_id=test_case.id,
                model=provider.model,
                provider=provider.name,
                prompt=test_case.prompt,
                response=response.content,
                expected=test_case.expected,
                metrics=metric_results,
                latency_ms=response.latency_ms,
                cost_usd=response.cost_usd,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.total_tokens,
                retry_count=retry_count,
                metadata={
                    "category": test_case.category,
                    "tags": test_case.tags,
                    "test_metadata": test_case.metadata,
                },
            )

        except Exception as e:
            # Retry if configured
            if retry_count < config.max_retries:
                logger.warning(f"Retrying test case {test_case.id} (attempt {retry_count + 1})")
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                return await self._evaluate_test_case(
                    provider,
                    test_case,
                    config,
                    retry_count + 1,
                )

            # Failed after retries
            logger.error(f"Test case {test_case.id} failed: {e}")
            return TestResult(
                test_id=test_case.id,
                model=provider.model,
                provider=provider.name,
                prompt=test_case.prompt,
                response="",
                expected=test_case.expected,
                metrics={},
                error=str(e),
                retry_count=retry_count,
            )

    def _generate_summary(
        self,
        results: list[TestResult],
        config: EvaluationConfig,
    ) -> dict[str, Any]:
        """Generate summary statistics."""
        if not results:
            return {
                "total_tests": 0,
                "successful_tests": 0,
                "failed_tests": 0,
                "models_evaluated": 0,
            }

        # Group by model
        by_model: dict[str, list[TestResult]] = {}
        for result in results:
            by_model.setdefault(result.model, []).append(result)

        # Calculate per-model stats
        model_summaries = {}
        for model, model_results in by_model.items():
            successful = [r for r in model_results if not r.error]
            failed = [r for r in model_results if r.error]

            # Metric averages
            metric_averages: dict[str, dict[str, float]] = {}
            for metric_name in config.metrics:
                values = [
                    r.metrics.get(metric_name, MetricResult(name=metric_name, value=0)).value
                    for r in successful
                    if metric_name in r.metrics
                ]
                if values:
                    metric_averages[metric_name] = {
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                    }

            model_summaries[model] = {
                "total_tests": len(model_results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": (len(successful) / len(model_results) * 100) if model_results else 0,
                "avg_latency_ms": sum(r.latency_ms for r in successful) / len(successful) if successful else 0,
                "total_cost_usd": sum(r.cost_usd for r in model_results),
                "avg_score": sum(r.overall_score for r in successful) / len(successful) if successful else 0,
                "metric_averages": metric_averages,
            }

        # Overall stats
        all_successful = [r for r in results if not r.error]
        all_failed = [r for r in results if r.error]

        return {
            "total_tests": len(results),
            "successful_tests": len(all_successful),
            "failed_tests": len(all_failed),
            "success_rate": (len(all_successful) / len(results) * 100) if results else 0,
            "total_cost_usd": sum(r.cost_usd for r in results),
            "total_tokens": sum(r.total_tokens for r in results),
            "avg_latency_ms": sum(r.latency_ms for r in all_successful) / len(all_successful) if all_successful else 0,
            "models_evaluated": len(by_model),
            "model_summaries": model_summaries,
            "metric_summary": self._generate_metric_summary(all_successful, config.metrics),
        }

    def _generate_metric_summary(
        self,
        results: list[TestResult],
        metric_names: list[str],
    ) -> dict[str, dict[str, float]]:
        """Generate summary for each metric across all results."""
        summary = {}

        for metric_name in metric_names:
            values = []
            passed = 0

            for result in results:
                if metric_name in result.metrics:
                    metric_result = result.metrics[metric_name]
                    values.append(metric_result.value)
                    if metric_result.passed:
                        passed += 1

            if values:
                summary[metric_name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "pass_rate": (passed / len(values) * 100) if values else 0,
                    "count": len(values),
                }

        return summary

    def _save_results(self, result: EvaluationResult) -> Path:
        """Save results to file."""
        timestamp = datetime.fromisoformat(result.start_time).strftime("%Y%m%d_%H%M%S")
        filename = f"{result.config.name}_{timestamp}"

        # Save in requested format
        if result.config.output_format == "json":
            path = self.output_dir / f"{filename}.json"
            self._save_json(result, path)
        elif result.config.output_format in ["yaml", "yml"]:
            path = self.output_dir / f"{filename}.yaml"
            self._save_yaml(result, path)
        else:
            # Default to JSON
            path = self.output_dir / f"{filename}.json"
            self._save_json(result, path)

        logger.info(f"Saved results to {path}")
        return path

    def _save_json(self, result: EvaluationResult, path: Path) -> None:
        """Save results as JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    def _save_yaml(self, result: EvaluationResult, path: Path) -> None:
        """Save results as YAML."""
        import yaml

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(result.to_dict(), f, default_flow_style=False, sort_keys=False)

    async def _default_progress_callback(
        self,
        completed: int,
        total: int,
        model: str,
    ) -> None:
        """Default progress callback (logs to console)."""
        percent = (completed / total * 100) if total > 0 else 100
        logger.info(f"Progress: {completed}/{total} ({percent:.1f}%) - Model: {model}")


# ============================================================================
# Convenience Functions
# ============================================================================

async def run_evaluation(
    name: str,
    dataset: str,
    models: list[dict[str, Any]],
    metrics: list[str],
    **kwargs: Any,
) -> EvaluationResult:
    """
    Convenience function to run an evaluation.

    Args:
        name: Evaluation name
        dataset: Dataset name
        models: List of model configs [{"provider": "openai", "model": "gpt-4o-mini"}]
        metrics: List of metric names
        **kwargs: Additional config parameters

    Returns:
        EvaluationResult

    Examples:
        >>> result = await run_evaluation(
        ...     name="my_eval",
        ...     dataset="qa_evaluation",
        ...     models=[{"provider": "openai", "model": "gpt-4o-mini"}],
        ...     metrics=["exact_match", "semantic_similarity"],
        ...     parallel=3
        ... )
        >>> print(result.summary["success_rate"])
    """
    # Convert dict models to ModelConfig
    model_configs = [ModelConfig(**m) for m in models]

    config = EvaluationConfig(
        name=name,
        dataset=dataset,
        models=model_configs,
        metrics=metrics,
        **kwargs,
    )

    runner = EvaluationRunner()
    return await runner.run(config)


def create_config(
    name: str,
    dataset: str,
    models: list[dict[str, Any]],
    metrics: list[str],
    **kwargs: Any,
) -> EvaluationConfig:
    """
    Create an EvaluationConfig.

    Args:
        name: Evaluation name
        dataset: Dataset name
        models: List of model configs
        metrics: List of metric names
        **kwargs: Additional config parameters

    Returns:
        EvaluationConfig
    """
    model_configs = [ModelConfig(**m) for m in models]
    return EvaluationConfig(
        name=name,
        dataset=dataset,
        models=model_configs,
        metrics=metrics,
        **kwargs,
    )


# Export main classes and functions
__all__ = [
    "ModelConfig",
    "EvaluationConfig",
    "TestResult",
    "EvaluationResult",
    "ProgressTracker",
    "EvaluationRunner",
    "run_evaluation",
    "create_config",
]
