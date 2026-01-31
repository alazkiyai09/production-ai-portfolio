"""
A/B testing and experiment orchestration for prompt optimization.

This module provides the experimental framework for running systematic
prompt variation tests with statistical analysis.
"""

from src.prompt_optimizer.experiments.ab_test import (
    # Enums
    TestMethod,
    MultipleComparisonCorrection,
    # Data classes
    TestResult,
    ABTestConfig,
    # Main class
    ABTester,
    create_ab_tester,
)

from src.prompt_optimizer.experiments.framework import (
    # Enums
    ExperimentStatus,
    # Data classes
    ExperimentConfig,
    VariantResult,
    ExperimentResult,
    # Main class
    ExperimentFramework,
    create_experiment_framework,
)

__all__ = [
    # A/B Testing
    "TestMethod",
    "MultipleComparisonCorrection",
    "TestResult",
    "ABTestConfig",
    "ABTester",
    "create_ab_tester",
    # Framework
    "ExperimentStatus",
    "ExperimentConfig",
    "VariantResult",
    "ExperimentResult",
    "ExperimentFramework",
    "create_experiment_framework",
]
