"""
Statistical analysis modules for prompt evaluation.

This module provides comprehensive statistical testing and correction
methods for analyzing prompt optimization experiments.
"""

from src.prompt_optimizer.statistics.tests import (
    # Enums
    DistributionType,
    OutlierMethod,
    # Data classes
    DistributionTest,
    OutlierResult,
    PowerAnalysis,
    # Main class
    StatisticalTests,
    create_statistical_tests,
)

from src.prompt_optimizer.statistics.corrections import (
    # Enums
    CorrectionMethod,
    # Data classes
    CorrectionResult,
    # Main class
    MultipleComparisonCorrection,
    # Functions
    correct_p_values,
)

__all__ = [
    # Tests module
    "DistributionType",
    "OutlierMethod",
    "DistributionTest",
    "OutlierResult",
    "PowerAnalysis",
    "StatisticalTests",
    "create_statistical_tests",
    # Corrections module
    "CorrectionMethod",
    "CorrectionResult",
    "MultipleComparisonCorrection",
    "correct_p_values",
]
