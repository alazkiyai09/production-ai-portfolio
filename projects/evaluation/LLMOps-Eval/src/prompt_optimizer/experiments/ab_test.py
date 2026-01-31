"""
A/B/n testing framework for prompt optimization.

This module provides statistical testing capabilities for comparing
prompt variations with proper significance testing, effect size calculation,
and multiple comparison correction.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Callable
from enum import Enum
import numpy as np
from scipy import stats
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Classes
# ============================================================================

class TestMethod(Enum):
    """Statistical test methods."""

    # Parametric tests
    T_TEST = "t_test"  # Independent samples t-test
    PAIRED_T_TEST = "paired_t_test"  # Paired t-test (same test cases)
    ANOVA = "anova"  # One-way ANOVA for >2 groups

    # Non-parametric tests
    MANN_WHITNEY = "mann_whitney"  # Mann-Whitney U test
    WILCOXON = "wilcoxon"  # Wilcoxon signed-rank test
    KRUSKAL_WALLIS = "kruskal_wallis"  # Kruskal-Wallis test

    # Proportion tests
    Z_TEST = "z_test"  # Z-test for proportions
    CHI_SQUARE = "chi_square"  # Chi-square test


class MultipleComparisonCorrection(Enum):
    """Methods for correcting multiple comparisons."""

    NONE = "none"  # No correction
    BONFERRONI = "bonferroni"  # Bonferroni correction (conservative)
    HOLM = "holm"  # Holm-Bonferroni (less conservative)
    BENJAMINI_HOCHBERG = "bh"  # Benjamini-Hochberg FDR
    BENJAMINI_YEKUTIELI = "by"  # Benjamini-Yekutieli (more conservative FDR)


@dataclass
class TestResult:
    """
    Result of a statistical test.

    Attributes:
        test_name: Name of the statistical test
        statistic: Test statistic value
        p_value: P-value of the test
        is_significant: Whether result is statistically significant
        effect_size: Effect size (Cohen's d, r, etc.)
        effect_size_magnitude: Interpretation of effect size
        confidence_interval: 95% confidence interval for effect size
        control_mean: Mean of control group
        treatment_mean: Mean of treatment group
        control_std: Std dev of control group
        treatment_std: Std dev of treatment group
        sample_size: Sample size per group
        power: Statistical power (1 - beta)
        metadata: Additional metadata
    """

    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    effect_size: float
    effect_size_magnitude: str  # "negligible", "small", "medium", "large"
    confidence_interval: Tuple[float, float]
    control_mean: float
    treatment_mean: float
    control_std: float
    treatment_std: float
    sample_size: int
    power: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "effect_size": self.effect_size,
            "effect_size_magnitude": self.effect_size_magnitude,
            "confidence_interval": self.confidence_interval,
            "control_mean": self.control_mean,
            "treatment_mean": self.treatment_mean,
            "control_std": self.control_std,
            "treatment_std": self.treatment_std,
            "sample_size": self.sample_size,
            "power": self.power,
            "metadata": self.metadata,
        }


@dataclass
class ABTestConfig:
    """
    Configuration for A/B testing.

    Attributes:
        alpha: Significance level (typically 0.05)
        power: Desired statistical power (typically 0.80)
        effect_size_threshold: Minimum detectable effect size
        correction_method: Multiple comparison correction method
        min_sample_size: Minimum sample size per variant
        test_method: Which statistical test to use
        paired_design: Whether to use paired tests (same test cases)
        metric_direction: Which direction is "better" ("higher" or "lower")
    """

    alpha: float = 0.05
    power: float = 0.80
    effect_size_threshold: float = 0.5  # Cohen's d = 0.5 = medium effect
    correction_method: MultipleComparisonCorrection = MultipleComparisonCorrection.BONFERRONI
    min_sample_size: int = 30
    test_method: TestMethod = TestMethod.T_TEST
    paired_design: bool = True  # Use paired tests by default (same test cases)
    metric_direction: str = "higher"  # For metrics like accuracy, higher is better

    def validate(self) -> None:
        """Validate configuration."""
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {self.alpha}")
        if not 0 < self.power < 1:
            raise ValueError(f"power must be between 0 and 1, got {self.power}")
        if self.min_sample_size < 2:
            raise ValueError(f"min_sample_size must be >= 2, got {self.min_sample_size}")
        if self.metric_direction not in ["higher", "lower"]:
            raise ValueError(f"metric_direction must be 'higher' or 'lower', got {self.metric_direction}")


# ============================================================================
# Statistical Test Functions
# ============================================================================

class ABTester:
    """
    A/B/n testing with statistical analysis.

    Performs hypothesis tests, calculates effect sizes, and applies
    multiple comparison corrections.
    """

    def __init__(self, config: ABTestConfig):
        """
        Initialize the A/B tester.

        Args:
            config: Test configuration
        """
        config.validate()
        self.config = config

    def compare_variants(
        self,
        control_scores: List[float],
        treatment_scores: List[float],
        variant_name: str = "treatment",
    ) -> TestResult:
        """
        Compare two variants using statistical testing.

        Args:
            control_scores: Scores from control variant (baseline)
            treatment_scores: Scores from treatment variant
            variant_name: Name of treatment variant

        Returns:
            TestResult with statistical analysis
        """
        if len(control_scores) < self.config.min_sample_size:
            logger.warning(
                f"Control sample size ({len(control_scores)}) < min ({self.config.min_sample_size})"
            )
        if len(treatment_scores) < self.config.min_sample_size:
            logger.warning(
                f"Treatment sample size ({len(treatment_scores)}) < min ({self.config.min_sample_size})"
            )

        # Convert to numpy arrays
        control = np.array(control_scores)
        treatment = np.array(treatment_scores)

        # Select test method
        if self.config.test_method == TestMethod.T_TEST:
            return self._t_test(control, treatment, variant_name)
        elif self.config.test_method == TestMethod.PAIRED_T_TEST:
            return self._paired_t_test(control, treatment, variant_name)
        elif self.config.test_method == TestMethod.MANN_WHITNEY:
            return self._mann_whitney(control, treatment, variant_name)
        elif self.config.test_method == TestMethod.WILCOXON:
            return self._wilcoxon(control, treatment, variant_name)
        elif self.config.test_method == TestMethod.Z_TEST:
            return self._z_test(control, treatment, variant_name)
        else:
            # Default to t-test
            return self._t_test(control, treatment, variant_name)

    def compare_multiple_variants(
        self,
        scores: Dict[str, List[float]],
        control_name: str = "control",
    ) -> Dict[str, TestResult]:
        """
        Compare multiple variants against a control.

        Applies multiple comparison correction.

        Args:
            scores: Dictionary of variant names to score lists
            control_name: Name of control variant

        Returns:
            Dictionary of variant names to TestResults
        """
        if control_name not in scores:
            raise ValueError(f"Control variant '{control_name}' not found in scores")

        control_scores = scores[control_name]
        results = {}
        p_values = []

        # First pass: run all tests
        for variant_name, variant_scores in scores.items():
            if variant_name == control_name:
                continue

            result = self.compare_variants(control_scores, variant_scores, variant_name)
            results[variant_name] = result
            p_values.append(result.p_value)

        # Apply multiple comparison correction
        if len(p_values) > 1 and self.config.correction_method != MultipleComparisonCorrection.NONE:
            corrected = self._apply_correction(p_values)

            # Update results with corrected p-values
            for i, (variant_name, result) in enumerate(results.items()):
                result.p_value = corrected[i]
                result.is_significant = result.p_value < self.config.alpha
                result.metadata["corrected_p_value"] = corrected[i]
                result.metadata["correction_method"] = self.config.correction_method.value

        return results

    def _t_test(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        variant_name: str,
    ) -> TestResult:
        """Independent samples t-test."""
        # Perform t-test
        statistic, p_value = stats.ttest_ind(treatment, control, equal_var=False)

        # Calculate effect size (Cohen's d)
        effect_size = self._cohens_d(control, treatment)
        effect_magnitude = self._interpret_effect_size(effect_size)

        # Calculate confidence interval for effect size
        ci = self._effect_size_ci(control, treatment)

        # Calculate power
        power = self._calculate_power(
            len(control),
            len(treatment),
            effect_size,
            alpha=self.config.alpha,
        )

        # Determine significance
        is_significant = p_value < self.config.alpha

        # Determine if improvement (depends on metric direction)
        if self.config.metric_direction == "higher":
            improvement = np.mean(treatment) > np.mean(control)
        else:
            improvement = np.mean(treatment) < np.mean(control)

        return TestResult(
            test_name="Independent t-test",
            statistic=statistic,
            p_value=p_value,
            is_significant=is_significant and improvement,
            effect_size=effect_size,
            effect_size_magnitude=effect_magnitude,
            confidence_interval=ci,
            control_mean=float(np.mean(control)),
            treatment_mean=float(np.mean(treatment)),
            control_std=float(np.std(control, ddof=1)),
            treatment_std=float(np.std(treatment, ddof=1)),
            sample_size=len(control),
            power=power,
            metadata={"variant": variant_name, "improvement": improvement},
        )

    def _paired_t_test(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        variant_name: str,
    ) -> TestResult:
        """Paired samples t-test."""
        if len(control) != len(treatment):
            raise ValueError("Control and treatment must have same length for paired test")

        # Perform paired t-test
        statistic, p_value = stats.ttest_rel(treatment, control)

        # Calculate effect size (Cohen's d for paired samples)
        differences = treatment - control
        effect_size = np.mean(differences) / np.std(differences, ddof=1)
        effect_magnitude = self._interpret_effect_size(abs(effect_size))

        # Calculate confidence interval
        se = np.std(differences, ddof=1) / np.sqrt(len(differences))
        ci = (
            float(np.mean(differences) - 1.96 * se),
            float(np.mean(differences) + 1.96 * se),
        )

        # Calculate power
        power = self._calculate_paired_power(
            len(differences),
            effect_size,
            alpha=self.config.alpha,
        )

        # Determine significance
        is_significant = p_value < self.config.alpha

        # Determine direction of improvement
        if self.config.metric_direction == "higher":
            improvement = np.mean(treatment) > np.mean(control)
        else:
            improvement = np.mean(treatment) < np.mean(control)

        return TestResult(
            test_name="Paired t-test",
            statistic=statistic,
            p_value=p_value,
            is_significant=is_significant and improvement,
            effect_size=effect_size,
            effect_size_magnitude=effect_magnitude,
            confidence_interval=ci,
            control_mean=float(np.mean(control)),
            treatment_mean=float(np.mean(treatment)),
            control_std=float(np.std(control, ddof=1)),
            treatment_std=float(np.std(treatment, ddof=1)),
            sample_size=len(control),
            power=power,
            metadata={"variant": variant_name, "improvement": improvement},
        )

    def _mann_whitney(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        variant_name: str,
    ) -> TestResult:
        """Mann-Whitney U test (non-parametric)."""
        # Perform Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(
            treatment, control, alternative="two-sided"
        )

        # Calculate effect size (r = Z / sqrt(N))
        n1, n2 = len(control), len(treatment)
        z_score = stats.norm.ppf(p_value / 2)  # Approximate
        effect_size = abs(z_score) / np.sqrt(n1 + n2)
        effect_magnitude = self._interpret_effect_size(effect_size)

        # Approximate CI
        ci = (effect_size * 0.5, effect_size * 1.5)

        # Determine significance
        is_significant = p_value < self.config.alpha

        # Determine direction of improvement
        if self.config.metric_direction == "higher":
            improvement = np.mean(treatment) > np.mean(control)
        else:
            improvement = np.mean(treatment) < np.mean(control)

        return TestResult(
            test_name="Mann-Whitney U test",
            statistic=statistic,
            p_value=p_value,
            is_significant=is_significant and improvement,
            effect_size=effect_size,
            effect_size_magnitude=effect_magnitude,
            confidence_interval=ci,
            control_mean=float(np.mean(control)),
            treatment_mean=float(np.mean(treatment)),
            control_std=float(np.std(control, ddof=1)),
            treatment_std=float(np.std(treatment, ddof=1)),
            sample_size=len(control),
            power=None,  # Power calculation not available for non-parametric
            metadata={"variant": variant_name, "improvement": improvement},
        )

    def _wilcoxon(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        variant_name: str,
    ) -> TestResult:
        """Wilcoxon signed-rank test (paired, non-parametric)."""
        if len(control) != len(treatment):
            raise ValueError("Control and treatment must have same length for paired test")

        # Perform Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(treatment, control)

        # Calculate effect size (r)
        n = len(control)
        effect_size = abs(statistic - n * (n + 1) / 4) / (n * (n + 1) / 4)
        effect_magnitude = self._interpret_effect_size(effect_size)

        # Approximate CI
        ci = (effect_size * 0.5, effect_size * 1.5)

        # Determine significance
        is_significant = p_value < self.config.alpha

        # Determine direction of improvement
        if self.config.metric_direction == "higher":
            improvement = np.mean(treatment) > np.mean(control)
        else:
            improvement = np.mean(treatment) < np.mean(control)

        return TestResult(
            test_name="Wilcoxon signed-rank test",
            statistic=statistic,
            p_value=p_value,
            is_significant=is_significant and improvement,
            effect_size=effect_size,
            effect_size_magnitude=effect_magnitude,
            confidence_interval=ci,
            control_mean=float(np.mean(control)),
            treatment_mean=float(np.mean(treatment)),
            control_std=float(np.std(control, ddof=1)),
            treatment_std=float(np.std(treatment, ddof=1)),
            sample_size=len(control),
            power=None,
            metadata={"variant": variant_name, "improvement": improvement},
        )

    def _z_test(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        variant_name: str,
    ) -> TestResult:
        """Z-test for proportions."""
        # Convert to proportions if needed
        p1 = np.mean(control)
        p2 = np.mean(treatment)
        n1 = len(control)
        n2 = len(treatment)

        # Pooled proportion
        p_pooled = (np.sum(control) + np.sum(treatment)) / (n1 + n2)

        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))

        # Z-score
        z_score = (p2 - p1) / se

        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        # Effect size (Cohen's h)
        effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
        effect_magnitude = self._interpret_effect_size(abs(effect_size))

        # Confidence interval
        se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        ci = (
            float((p2 - p1) - 1.96 * se_diff),
            float((p2 - p1) + 1.96 * se_diff),
        )

        # Determine significance
        is_significant = p_value < self.config.alpha

        # Determine direction of improvement
        if self.config.metric_direction == "higher":
            improvement = p2 > p1
        else:
            improvement = p2 < p1

        return TestResult(
            test_name="Z-test for proportions",
            statistic=z_score,
            p_value=p_value,
            is_significant=is_significant and improvement,
            effect_size=effect_size,
            effect_size_magnitude=effect_magnitude,
            confidence_interval=ci,
            control_mean=float(p1),
            treatment_mean=float(p2),
            control_std=float(np.std(control, ddof=1)),
            treatment_std=float(np.std(treatment, ddof=1)),
            sample_size=len(control),
            power=None,
            metadata={"variant": variant_name, "improvement": improvement},
        )

    def _cohens_d(self, control: np.ndarray, treatment: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(control), len(treatment)
        var1, var2 = np.var(control, ddof=1), np.var(treatment, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        # Cohen's d
        if pooled_std == 0:
            return 0.0
        return (np.mean(treatment) - np.mean(control)) / pooled_std

    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude."""
        abs_es = abs(effect_size)
        if abs_es < 0.2:
            return "negligible"
        elif abs_es < 0.5:
            return "small"
        elif abs_es < 0.8:
            return "medium"
        else:
            return "large"

    def _effect_size_ci(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Calculate confidence interval for effect size."""
        n1, n2 = len(control), len(treatment)
        effect_size = self._cohens_d(control, treatment)

        # Standard error of effect size
        se = np.sqrt((n1 + n2) / (n1 * n2) + effect_size**2 / (2 * (n1 + n2)))

        # Z-score for confidence level
        z = stats.norm.ppf(1 - (1 - confidence) / 2)

        # CI
        lower = effect_size - z * se
        upper = effect_size + z * se

        return (float(lower), float(upper))

    def _calculate_power(
        self,
        n1: int,
        n2: int,
        effect_size: float,
        alpha: float = 0.05,
    ) -> float:
        """Calculate statistical power."""
        from scipy.stats import norm

        # Z-critical
        z_crit = norm.ppf(1 - alpha / 2)

        # Non-centrality parameter
        ncp = effect_size / np.sqrt(1/n1 + 1/n2)

        # Power (simplified)
        # P(Z > z_crit - ncp) + P(Z < -z_crit - ncp)
        power = (
            1 - norm.cdf(z_crit - ncp) +
            norm.cdf(-z_crit - ncp)
        )

        return float(min(max(power, 0.0), 1.0))

    def _calculate_paired_power(
        self,
        n: int,
        effect_size: float,
        alpha: float = 0.05,
    ) -> float:
        """Calculate power for paired test."""
        from scipy.stats import norm

        z_crit = norm.ppf(1 - alpha / 2)
        ncp = effect_size * np.sqrt(n)

        power = (
            1 - norm.cdf(z_crit - ncp) +
            norm.cdf(-z_crit - ncp)
        )

        return float(min(max(power, 0.0), 1.0))

    def _apply_correction(self, p_values: List[float]) -> List[float]:
        """Apply multiple comparison correction."""
        n = len(p_values)

        if self.config.correction_method == MultipleComparisonCorrection.BONFERRONI:
            # Bonferroni: multiply each p-value by n
            corrected = [min(p * n, 1.0) for p in p_values]

        elif self.config.correction_method == MultipleComparisonCorrection.HOLM:
            # Holm-Bonferroni: step-down procedure
            sorted_indices = sorted(range(n), key=lambda i: p_values[i])
            corrected = [1.0] * n

            for rank, idx in enumerate(sorted_indices):
                corrected[idx] = min(p_values[idx] * (n - rank), 1.0)

        elif self.config.correction_method == MultipleComparisonCorrection.BENJAMINI_HOCHBERG:
            # Benjamini-Hochberg FDR
            sorted_indices = sorted(range(n), key=lambda i: p_values[i])
            corrected = [1.0] * n

            for rank, idx in enumerate(sorted_indices):
                corrected[idx] = min(p_values[idx] * n / (rank + 1), 1.0)

        elif self.config.correction_method == MultipleComparisonCorrection.BENJAMINI_YEKUTIELI:
            # Benjamini-Yekutieli (more conservative)
            harmonic_sum = sum(1 / (i + 1) for i in range(n))
            sorted_indices = sorted(range(n), key=lambda i: p_values[i])
            corrected = [1.0] * n

            for rank, idx in enumerate(sorted_indices):
                corrected[idx] = min(
                    p_values[idx] * n / ((rank + 1) * harmonic_sum),
                    1.0
                )

        else:
            # No correction
            corrected = p_values

        return corrected

    def calculate_required_sample_size(
        self,
        effect_size: Optional[float] = None,
    ) -> int:
        """
        Calculate required sample size per variant.

        Uses power analysis with configured alpha, power, and effect size.

        Args:
            effect_size: Target effect size (defaults to config threshold)

        Returns:
            Required sample size per variant
        """
        if effect_size is None:
            effect_size = self.config.effect_size_threshold

        from scipy.stats import norm

        # Z-values
        z_alpha = norm.ppf(1 - self.config.alpha / 2)
        z_beta = norm.ppf(self.config.power)

        # Sample size formula for two-sample t-test
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        return int(np.ceil(n))

    def calculate_min_detectable_effect(
        self,
        sample_size: int,
    ) -> float:
        """
        Calculate minimum detectable effect size for a given sample size.

        Args:
            sample_size: Sample size per variant

        Returns:
            Minimum detectable effect size (Cohen's d)
        """
        from scipy.stats import norm

        z_alpha = norm.ppf(1 - self.config.alpha / 2)
        z_beta = norm.ppf(self.config.power)

        effect_size = (z_alpha + z_beta) * np.sqrt(2 / sample_size)

        return float(effect_size)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_ab_tester(
    alpha: float = 0.05,
    power: float = 0.80,
    effect_size_threshold: float = 0.5,
    correction_method: MultipleComparisonCorrection = MultipleComparisonCorrection.BONFERRONI,
    min_sample_size: int = 30,
    test_method: TestMethod = TestMethod.T_TEST,
    paired_design: bool = True,
    metric_direction: str = "higher",
) -> ABTester:
    """
    Factory function to create an ABTester.

    Args:
        alpha: Significance level
        power: Statistical power
        effect_size_threshold: Minimum detectable effect
        correction_method: Multiple comparison correction
        min_sample_size: Minimum sample size
        test_method: Statistical test to use
        paired_design: Whether to use paired tests
        metric_direction: Which direction is "better"

    Returns:
        Configured ABTester
    """
    config = ABTestConfig(
        alpha=alpha,
        power=power,
        effect_size_threshold=effect_size_threshold,
        correction_method=correction_method,
        min_sample_size=min_sample_size,
        test_method=test_method,
        paired_design=paired_design,
        metric_direction=metric_direction,
    )

    return ABTester(config)


# Export main classes and functions
__all__ = [
    # Enums
    "TestMethod",
    "MultipleComparisonCorrection",
    # Data classes
    "TestResult",
    "ABTestConfig",
    # Main class
    "ABTester",
    "create_ab_tester",
]
