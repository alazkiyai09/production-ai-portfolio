"""
Statistical tests for prompt evaluation.

This module provides comprehensive statistical testing capabilities
for analyzing prompt performance with proper significance testing.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class DistributionType(Enum):
    """Types of distributions."""

    NORMAL = "normal"
    LOG_NORMAL = "log_normal"
    EXPONENTIAL = "exponential"
    UNIFORM = "uniform"
    UNKNOWN = "unknown"


class OutlierMethod(Enum):
    """Methods for detecting outliers."""

    IQR = "iqr"  # Interquartile range
    Z_SCORE = "z_score"  # Z-score method
    ISOLATION_FOREST = "isolation_forest"  # Isolation forest
    DBSCAN = "dbscan"  # DBSCAN clustering


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DistributionTest:
    """Result of distribution normality test."""

    is_normal: bool
    test_name: str
    statistic: float
    p_value: float
    distribution_type: DistributionType
    confidence: float


@dataclass
class OutlierResult:
    """Result of outlier detection."""

    outlier_indices: List[int]
    outlier_values: List[float]
    method: OutlierMethod
    threshold: float
    cleaned_data: np.ndarray
    outlier_fraction: float


@dataclass
class PowerAnalysis:
    """Result of power analysis."""

    required_sample_size: int
    achieved_power: float
    effect_size: float
    alpha: float
    minimum_detectable_effect: float


# ============================================================================
# Statistical Test Functions
# ============================================================================

class StatisticalTests:
    """
    Comprehensive statistical testing for prompt evaluation.

    Provides normality tests, outlier detection, power analysis,
    and various hypothesis tests.
    """

    @staticmethod
    def test_normality(
        data: np.ndarray,
        alpha: float = 0.05,
        method: str = "shapiro",
    ) -> DistributionTest:
        """
        Test if data is normally distributed.

        Args:
            data: Data to test
            alpha: Significance level
            method: Test method ("shapiro", "ks", "anderson", "dagostino")

        Returns:
            DistributionTest result
        """
        if len(data) < 3:
            return DistributionTest(
                is_normal=False,
                test_name="insufficient_data",
                statistic=0.0,
                p_value=1.0,
                distribution_type=DistributionType.UNKNOWN,
                confidence=0.0,
            )

        if method == "shapiro":
            # Shapiro-Wilk test (recommended for n < 5000)
            statistic, p_value = stats.shapiro(data)
            test_name = "Shapiro-Wilk"

        elif method == "ks":
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.kstest(data, "norm")
            test_name = "Kolmogorov-Smirnov"

        elif method == "anderson":
            # Anderson-Darling test
            result = stats.anderson(data, dist="norm")
            # Compare to critical value at alpha
            critical_value = result.critical_values[2]  # 5% significance
            statistic = result.statistic
            p_value = 0.05 if statistic < critical_value else 0.01
            test_name = "Anderson-Darling"

        elif method == "dagostino":
            # D'Agostino's K-squared test
            statistic, p_value = stats.normaltest(data)
            test_name = "D'Agostino K-squared"

        else:
            raise ValueError(f"Unknown normality test method: {method}")

        is_normal = p_value > alpha

        # Determine distribution type
        if is_normal:
            dist_type = DistributionType.NORMAL
        else:
            # Check for log-normal
            log_data = np.log(data[data > 0])
            if len(log_data) > 3:
                _, log_p = stats.shapiro(log_data)
                dist_type = DistributionType.LOG_NORMAL if log_p > alpha else DistributionType.UNKNOWN
            else:
                dist_type = DistributionType.UNKNOWN

        return DistributionTest(
            is_normal=is_normal,
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            distribution_type=dist_type,
            confidence=1 - alpha,
        )

    @staticmethod
    def detect_outliers(
        data: np.ndarray,
        method: OutlierMethod = OutlierMethod.IQR,
        threshold: float = 1.5,
    ) -> OutlierResult:
        """
        Detect outliers in data.

        Args:
            data: Data to analyze
            method: Detection method
            threshold: Threshold for outlier detection

        Returns:
            OutlierResult with detected outliers
        """
        data = np.array(data)

        if method == OutlierMethod.IQR:
            # Interquartile range method
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1

            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            outlier_mask = (data < lower_bound) | (data > upper_bound)

        elif method == OutlierMethod.Z_SCORE:
            # Z-score method
            mean = np.mean(data)
            std = np.std(data)
            z_scores = np.abs((data - mean) / std)

            outlier_mask = z_scores > threshold

        elif method == OutlierMethod.ISOLATION_FOREST:
            # Isolation forest
            try:
                from sklearn.ensemble import IsolationForest

                clf = IsolationForest(contamination=0.1, random_state=42)
                outlier_mask = clf.fit_predict(data.reshape(-1, 1)) == -1
            except ImportError:
                logger.warning("sklearn not available, falling back to IQR method")
                return StatisticalTests.detect_outliers(data, OutlierMethod.IQR, threshold)

        elif method == OutlierMethod.DBSCAN:
            # DBSCAN clustering
            try:
                from sklearn.cluster import DBSCAN

                clustering = DBSCAN(eps=threshold, min_samples=3)
                outlier_mask = clustering.fit_predict(data.reshape(-1, 1)) == -1
            except ImportError:
                logger.warning("sklearn not available, falling back to IQR method")
                return StatisticalTests.detect_outliers(data, OutlierMethod.IQR, threshold)

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_values = data[outlier_mask].tolist()
        cleaned_data = data[~outlier_mask]

        return OutlierResult(
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            method=method,
            threshold=threshold,
            cleaned_data=cleaned_data,
            outlier_fraction=len(outlier_indices) / len(data),
        )

    @staticmethod
    def power_analysis(
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.80,
        ratio: float = 1.0,
    ) -> PowerAnalysis:
        """
        Calculate required sample size for given power and effect size.

        Args:
            effect_size: Cohen's d effect size
            alpha: Significance level
            power: Desired power (1 - beta)
            ratio: Ratio of sample sizes (n2/n1)

        Returns:
            PowerAnalysis result
        """
        from scipy.stats import norm

        # Z-values
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)

        # Sample size formula for two-sample t-test
        n1 = ((z_alpha + z_beta) / effect_size) ** 2 * (1 + ratio) / ratio
        n1 = int(np.ceil(n1))

        # Calculate achieved power
        ncp = effect_size / np.sqrt(1/n1 + ratio/n1)
        achieved_power = float(
            1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
        )
        achieved_power = min(max(achieved_power, 0.0), 1.0)

        # Minimum detectable effect for this sample size
        mde = (z_alpha + z_beta) * np.sqrt((1 + ratio) / (n1 * ratio))

        return PowerAnalysis(
            required_sample_size=n1,
            achieved_power=achieved_power,
            effect_size=effect_size,
            alpha=alpha,
            minimum_detectable_effect=mde,
        )

    @staticmethod
    def calculate_effect_size(
        control: np.ndarray,
        treatment: np.ndarray,
        method: str = "cohens_d",
    ) -> float:
        """
        Calculate effect size between two groups.

        Args:
            control: Control group scores
            treatment: Treatment group scores
            method: Effect size method ("cohens_d", "hedges_g", "glass_delta", "r")

        Returns:
            Effect size value
        """
        control = np.array(control)
        treatment = np.array(treatment)

        n1, n2 = len(control), len(treatment)
        mean1, mean2 = np.mean(control), np.mean(treatment)

        if method == "cohens_d":
            # Cohen's d (pooled SD)
            var1, var2 = np.var(control, ddof=1), np.var(treatment, ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            if pooled_std == 0:
                return 0.0
            return (mean2 - mean1) / pooled_std

        elif method == "hedges_g":
            # Hedges' g (bias-corrected Cohen's d)
            var1, var2 = np.var(control, ddof=1), np.var(treatment, ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            if pooled_std == 0:
                return 0.0

            # Bias correction factor
            J = 1 - 3 / (4 * (n1 + n2) - 9)
            return J * (mean2 - mean1) / pooled_std

        elif method == "glass_delta":
            # Glass' delta (using control SD)
            std1 = np.std(control, ddof=1)
            if std1 == 0:
                return 0.0
            return (mean2 - mean1) / std1

        elif method == "r":
            # Correlation coefficient effect size
            # From t-statistic
            t_stat, _ = stats.ttest_ind(treatment, control)
            return np.sqrt(t_stat ** 2 / (t_stat ** 2 + n1 + n2 - 2))

        else:
            raise ValueError(f"Unknown effect size method: {method}")

    @staticmethod
    def interpret_effect_size(effect_size: float, method: str = "cohens_d") -> str:
        """
        Interpret effect size magnitude.

        Args:
            effect_size: Effect size value
            method: Effect size method

        Returns:
            Interpretation string
        """
        abs_es = abs(effect_size)

        if method == "cohens_d":
            if abs_es < 0.2:
                return "negligible"
            elif abs_es < 0.5:
                return "small"
            elif abs_es < 0.8:
                return "medium"
            else:
                return "large"

        elif method == "r":
            if abs_es < 0.1:
                return "negligible"
            elif abs_es < 0.3:
                return "small"
            elif abs_es < 0.5:
                return "medium"
            else:
                return "large"

        else:
            return f"effect_size={effect_size:.3f}"

    @staticmethod
    def calculate_confidence_interval(
        data: np.ndarray,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for mean.

        Args:
            data: Data to analyze
            confidence: Confidence level (e.g., 0.95)

        Returns:
            (lower_bound, upper_bound)
        """
        data = np.array(data)
        n = len(data)
        mean = np.mean(data)
        std_err = stats.sem(data)

        # T-critical value
        t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)

        margin = t_crit * std_err

        return (float(mean - margin), float(mean + margin))

    @staticmethod
    def equivalence_test(
        control: np.ndarray,
        treatment: np.ndarray,
        margin: float,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Test for equivalence (not just difference).

        Tests if treatment is within margin of control.

        Args:
            control: Control group scores
            treatment: Treatment group scores
            margin: Equivalence margin
            alpha: Significance level

        Returns:
            Dictionary with test results
        """
        control = np.array(control)
        treatment = np.array(treatment)

        # Two one-sided tests (TOST)
        diff = np.mean(treatment) - np.mean(control)
        n1, n2 = len(control), len(treatment)

        # Pooled standard error
        se = np.sqrt(
            np.var(control, ddof=1) / n1 +
            np.var(treatment, ddof=1) / n2
        )

        # TOST statistics
        t1 = (diff - margin) / se
        t2 = (diff + margin) / se

        # P-values (one-sided)
        df = n1 + n2 - 2
        p1 = 1 - stats.t.cdf(t1, df)
        p2 = stats.t.cdf(t2, df)

        # Overall p-value
        p_value = max(p1, p2)

        # Is equivalent?
        is_equivalent = p_value < alpha and abs(diff) < margin

        return {
            "is_equivalent": is_equivalent,
            "p_value": p_value,
            "difference": diff,
            "margin": margin,
            "lower_bound": diff - margin,
            "upper_bound": diff + margin,
            "tost_p1": p1,
            "tost_p2": p2,
        }

    @staticmethod
    def non_inferiority_test(
        control: np.ndarray,
        treatment: np.ndarray,
        margin: float,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Test for non-inferiority.

        Tests if treatment is not worse than control by more than margin.

        Args:
            control: Control group scores
            treatment: Treatment group scores
            margin: Non-inferiority margin
            alpha: Significance level

        Returns:
            Dictionary with test results
        """
        control = np.array(control)
        treatment = np.array(treatment)

        # Difference (treatment - control)
        diff = np.mean(treatment) - np.mean(control)
        n1, n2 = len(control), len(treatment)

        # Standard error
        se = np.sqrt(
            np.var(control, ddof=1) / n1 +
            np.var(treatment, ddof=1) / n2
        )

        # T-statistic
        t_stat = (diff + margin) / se

        # P-value (one-sided)
        df = n1 + n2 - 2
        p_value = 1 - stats.t.cdf(t_stat, df)

        # Is non-inferior?
        is_non_inferior = p_value < alpha and diff > -margin

        return {
            "is_non_inferior": is_non_inferior,
            "p_value": p_value,
            "difference": diff,
            "margin": margin,
            "lower_bound": diff - margin,
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_statistical_tests() -> StatisticalTests:
    """Factory function to create StatisticalTests instance."""
    return StatisticalTests()


# Export
__all__ = [
    # Enums
    "DistributionType",
    "OutlierMethod",
    # Data classes
    "DistributionTest",
    "OutlierResult",
    "PowerAnalysis",
    # Main class
    "StatisticalTests",
    "create_statistical_tests",
]
