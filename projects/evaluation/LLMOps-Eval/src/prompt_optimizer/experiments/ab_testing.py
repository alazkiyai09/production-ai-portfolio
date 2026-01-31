"""
A/B testing framework for prompt experiments.

This module provides a framework for running A/B tests on prompts with
proper experimental design, random assignment, sample size calculation,
and results collection.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
import random
import uuid
import json
from pathlib import Path


class ExperimentStatus(Enum):
    """Status of an A/B test experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentVariant:
    """A variant in an A/B test."""
    id: str
    name: str
    prompt_variation: 'PromptVariation'
    is_control: bool = False


@dataclass
class ExperimentResult:
    """Result of a single test case in an experiment."""
    variant_id: str
    test_case_id: str
    response: str
    metrics: Dict[str, float]
    latency_ms: float
    cost_usd: float
    timestamp: str
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class Experiment:
    """An A/B test experiment."""
    id: str
    name: str
    description: str
    variants: List[ExperimentVariant]
    test_dataset_id: str
    metrics: List[str]
    status: ExperimentStatus
    config: Dict[str, Any]
    results: List[ExperimentResult] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    @property
    def sample_size_per_variant(self) -> int:
        """Get current sample size per variant."""
        counts = {}
        for result in self.results:
            counts[result.variant_id] = counts.get(result.variant_id, 0) + 1
        return min(counts.values()) if counts else 0


class ABTestingFramework:
    """Framework for running A/B tests on prompts."""

    def __init__(
        self,
        storage_path: str = "./experiments",
        evaluator: 'EvaluationRunner' = None
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.evaluator = evaluator
        self._experiments: Dict[str, Experiment] = {}
        self._load_experiments()

    def create_experiment(
        self,
        name: str,
        description: str,
        control_prompt: 'PromptVariation',
        treatment_prompts: List['PromptVariation'],
        test_dataset_id: str,
        metrics: List[str],
        config: Optional[Dict[str, Any]] = None
    ) -> Experiment:
        """
        Create a new A/B test experiment.

        Args:
            name: Experiment name
            description: What we're testing
            control_prompt: The baseline prompt (A)
            treatment_prompts: The variations to test (B, C, ...)
            test_dataset_id: Dataset to use for testing
            metrics: Which metrics to evaluate
            config: Additional configuration

        Returns:
            Created Experiment
        """
        experiment_id = str(uuid.uuid4())[:8]

        # Create variants
        variants = [
            ExperimentVariant(
                id=f"{experiment_id}_control",
                name="Control",
                prompt_variation=control_prompt,
                is_control=True
            )
        ]

        for i, treatment in enumerate(treatment_prompts):
            variants.append(ExperimentVariant(
                id=f"{experiment_id}_treatment_{i}",
                name=f"Treatment {i+1}",
                prompt_variation=treatment,
                is_control=False
            ))

        experiment = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            variants=variants,
            test_dataset_id=test_dataset_id,
            metrics=metrics,
            status=ExperimentStatus.DRAFT,
            config=config or {
                "min_sample_size": 30,
                "significance_level": 0.05,
                "random_seed": 42
            }
        )

        self._experiments[experiment_id] = experiment
        self._save_experiment(experiment)

        return experiment

    def start_experiment(self, experiment_id: str) -> Experiment:
        """Start an experiment."""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        if experiment.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Cannot start experiment in status: {experiment.status}")

        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.utcnow().isoformat()
        self._save_experiment(experiment)

        return experiment

    async def run_experiment(
        self,
        experiment_id: str,
        test_cases: List['TestCase'],
        llm_provider: 'LLMProvider'
    ) -> Experiment:
        """
        Run an experiment to completion.

        Args:
            experiment_id: Experiment to run
            test_cases: Test cases from the dataset
            llm_provider: LLM provider to use

        Returns:
            Completed experiment with results
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        if experiment.status != ExperimentStatus.RUNNING:
            self.start_experiment(experiment_id)
            experiment = self._experiments[experiment_id]

        min_samples = experiment.config.get("min_sample_size", 30)
        random.seed(experiment.config.get("random_seed", 42))

        # Run until we have enough samples
        for test_case in test_cases:
            # Random assignment to variant
            variant = random.choice(experiment.variants)

            # Execute
            result = await self._execute_variant(
                variant,
                test_case,
                llm_provider,
                experiment.metrics
            )

            experiment.results.append(result)

            # Check if we have enough samples
            if experiment.sample_size_per_variant >= min_samples:
                break

        # Check completion
        if experiment.sample_size_per_variant >= min_samples:
            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_at = datetime.utcnow().isoformat()

        self._save_experiment(experiment)
        return experiment

    async def _execute_variant(
        self,
        variant: ExperimentVariant,
        test_case: 'TestCase',
        llm_provider: 'LLMProvider',
        metrics: List[str]
    ) -> ExperimentResult:
        """Execute a single variant on a test case."""
        import time

        prompt = variant.prompt_variation.prompt_content
        system_prompt = variant.prompt_variation.system_prompt

        start_time = time.perf_counter()
        response = await llm_provider.generate(
            prompt=prompt,
            system_prompt=system_prompt
        )
        latency = (time.perf_counter() - start_time) * 1000

        # Evaluate metrics
        metric_results = {}
        if self.evaluator:
            from src.evaluation.metrics import create_metric
            for metric_name in metrics:
                metric = create_metric(metric_name)
                result = await metric.evaluate(
                    response.content,
                    test_case.expected,
                    test_case.context
                )
                metric_results[metric_name] = result.value

        return ExperimentResult(
            variant_id=variant.id,
            test_case_id=test_case.id,
            response=response.content,
            metrics=metric_results,
            latency_ms=latency,
            cost_usd=response.cost_usd,
            timestamp=datetime.utcnow().isoformat(),
            raw_response={"tokens": response.total_tokens}
        )

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        return self._experiments.get(experiment_id)

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None
    ) -> List[Experiment]:
        """List all experiments, optionally filtered by status."""
        experiments = list(self._experiments.values())
        if status:
            experiments = [e for e in experiments if e.status == status]
        return sorted(experiments, key=lambda e: e.created_at, reverse=True)

    def pause_experiment(self, experiment_id: str) -> Experiment:
        """Pause a running experiment."""
        experiment = self._experiments.get(experiment_id)
        if experiment and experiment.status == ExperimentStatus.RUNNING:
            experiment.status = ExperimentStatus.PAUSED
            self._save_experiment(experiment)
        return experiment

    def cancel_experiment(self, experiment_id: str) -> Experiment:
        """Cancel an experiment."""
        experiment = self._experiments.get(experiment_id)
        if experiment:
            experiment.status = ExperimentStatus.CANCELLED
            self._save_experiment(experiment)
        return experiment

    def _save_experiment(self, experiment: Experiment):
        """Save experiment to disk."""
        file_path = self.storage_path / f"{experiment.id}.json"
        with open(file_path, 'w') as f:
            # Convert to serializable format
            data = {
                "id": experiment.id,
                "name": experiment.name,
                "description": experiment.description,
                "variants": [
                    {
                        "id": v.id,
                        "name": v.name,
                        "is_control": v.is_control,
                        "prompt_content": v.prompt_variation.prompt_content,
                        "system_prompt": v.prompt_variation.system_prompt
                    }
                    for v in experiment.variants
                ],
                "test_dataset_id": experiment.test_dataset_id,
                "metrics": experiment.metrics,
                "status": experiment.status.value,
                "config": experiment.config,
                "results": [
                    {
                        "variant_id": r.variant_id,
                        "test_case_id": r.test_case_id,
                        "response": r.response,
                        "metrics": r.metrics,
                        "latency_ms": r.latency_ms,
                        "cost_usd": r.cost_usd,
                        "timestamp": r.timestamp
                    }
                    for r in experiment.results
                ],
                "created_at": experiment.created_at,
                "started_at": experiment.started_at,
                "completed_at": experiment.completed_at
            }
            json.dump(data, f, indent=2)

    def _load_experiments(self):
        """Load all experiments from disk."""
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    # Reconstruct experiment (simplified)
                    experiment = Experiment(
                        id=data["id"],
                        name=data["name"],
                        description=data["description"],
                        variants=[],  # Would need full reconstruction
                        test_dataset_id=data["test_dataset_id"],
                        metrics=data["metrics"],
                        status=ExperimentStatus(data["status"]),
                        config=data["config"],
                        created_at=data["created_at"],
                        started_at=data.get("started_at"),
                        completed_at=data.get("completed_at")
                    )
                    self._experiments[experiment.id] = experiment
            except Exception as e:
                print(f"Failed to load experiment {file_path}: {e}")


# Export main classes
__all__ = [
    "ExperimentStatus",
    "ExperimentVariant",
    "ExperimentResult",
    "Experiment",
    "ABTestingFramework",
]
