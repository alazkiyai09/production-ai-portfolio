"""
Multiple comparison correction methods.

This module provides various methods for correcting p-values when
performing multiple statistical tests.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class CorrectionMethod(Enum):
    """Multiple comparison correction methods."""

    NONE = "none"  # No correction
    BONFERRONI = "bonferroni"  # Bonferroni (simple, conservative)
    HOLM = "holm"  # Holm-Bonferroni (step-down, less conservative)
    HOCHBERG = "hochberg"  # Hochberg (step-up)
    HOMMEL = "hommel"  # Hommel (more powerful than Hochberg)
    SIDAK = "sidak"  # Sidak correction
    BENJAMINI_HOCHBERG = "bh"  # Benjamini-Hochberg FDR
    BENJAMINI_YEKUTIELI = "by"  # Benjamini-Yekutieli (conservative FDR)
    STOREY = "storey"  # Storey's q-value


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CorrectionResult:
    """
    Result of multiple comparison correction.

    Attributes:
        original_p_values: Original p-values
        corrected_p_values: Adjusted p-values
        rejected: Boolean array of rejected hypotheses
        method: Correction method used
        alpha: Significance level
        num_rejections: Number of rejected hypotheses
        num_tests: Total number of tests
        false_discovery_rate: Estimated FDR
    """

    original_p_values: List[float]
    corrected_p_values: List[float]
    rejected: List[bool]
    method: CorrectionMethod
    alpha: float
    num_rejections: int
    num_tests: int
    false_discovery_rate: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "original_p_values": self.original_p_values,
            "corrected_p_values": self.corrected_p_values,
            "rejected": self.rejected,
            "method": self.method.value,
            "alpha": self.alpha,
            "num_rejections": self.num_rejections,
            "num_tests": self.num_tests,
            "false_discovery_rate": self.false_discovery_rate,
        }


# ============================================================================
# Correction Methods
# ============================================================================

class MultipleComparisonCorrection:
    """
    Apply multiple comparison corrections to p-values.

    Controls family-wise error rate (FWER) or false discovery rate (FDR).
    """

    @staticmethod
    def bonferroni(p_values: List[float], alpha: float = 0.05) -> CorrectionResult:
        """
        Bonferroni correction.

        Most conservative: multiply each p-value by number of tests.

        Args:
            p_values: List of p-values
            alpha: Significance level

        Returns:
            CorrectionResult
        """
        n = len(p_values)
        corrected = [min(p * n, 1.0) for p in p_values]
        rejected = [p < alpha for p in corrected]

        return CorrectionResult(
            original_p_values=p_values,
            corrected_p_values=corrected,
            rejected=rejected,
            method=CorrectionMethod.BONFERRONI,
            alpha=alpha,
            num_rejections=sum(rejected),
            num_tests=n,
        )

    @staticmethod
    def holm(p_values: List[float], alpha: float = 0.05) -> CorrectionResult:
        """
        Holm-Bonferroni correction (step-down).

        Less conservative than Bonferroni while controlling FWER.

        Args:
            p_values: List of p-values
            alpha: Significance level

        Returns:
            CorrectionResult
        """
        n = len(p_values)

        # Sort p-values with indices
        sorted_indices = sorted(range(n), key=lambda i: p_values[i])
        corrected = [1.0] * n

        # Step-down procedure
        for rank, idx in enumerate(sorted_indices):
            correction_factor = n - rank
            corrected[idx] = min(p_values[idx] * correction_factor, 1.0)

        rejected = [p < alpha for p in corrected]

        return CorrectionResult(
            original_p_values=p_values,
            corrected_p_values=corrected,
            rejected=rejected,
            method=CorrectionMethod.HOLM,
            alpha=alpha,
            num_rejections=sum(rejected),
            num_tests=n,
        )

    @staticmethod
    def hochberg(p_values: List[float], alpha: float = 0.05) -> CorrectionResult:
        """
        Hochberg correction (step-up).

        More powerful than Holm for independent tests.

        Args:
            p_values: List of p-values
            alpha: Significance level

        Returns:
            CorrectionResult
        """
        n = len(p_values)

        # Sort p-values in descending order with indices
        sorted_indices = sorted(range(n), key=lambda i: -p_values[i])
        corrected = [1.0] * n

        # Step-up procedure
        for rank, idx in enumerate(sorted_indices):
            correction_factor = rank + 1
            corrected[idx] = min(p_values[idx] * correction_factor, 1.0)

        rejected = [p < alpha for p in corrected]

        return CorrectionResult(
            original_p_values=p_values,
            corrected_p_values=corrected,
            rejected=rejected,
            method=CorrectionMethod.HOCHBERG,
            alpha=alpha,
            num_rejections=sum(rejected),
            num_tests=n,
        )

    @staticmethod
    def sidak(p_values: List[float], alpha: float = 0.05) -> CorrectionResult:
        """
        Sidak correction.

        Slightly less conservative than Bonferroni.

        Args:
            p_values: List of p-values
            alpha: Significance level

        Returns:
            CorrectionResult
        """
        n = len(p_values)
        # Sidak correction: 1 - (1 - alpha)^(1/n)
        corrected = [1 - (1 - p) ** n for p in p_values]
        corrected = [min(p, 1.0) for p in corrected]

        rejected = [p < alpha for p in corrected]

        return CorrectionResult(
            original_p_values=p_values,
            corrected_p_values=corrected,
            rejected=rejected,
            method=CorrectionMethod.SIDAK,
            alpha=alpha,
            num_rejections=sum(rejected),
            num_tests=n,
        )

    @staticmethod
    def benjamini_hochberg(
        p_values: List[float],
        alpha: float = 0.05,
    ) -> CorrectionResult:
        """
        Benjamini-Hochberg FDR correction.

        Controls false discovery rate rather than FWER.
        Less conservative, more powerful.

        Args:
            p_values: List of p-values
            alpha: Significance level (FDR level)

        Returns:
            CorrectionResult
        """
        n = len(p_values)

        # Sort p-values with indices
        sorted_indices = sorted(range(n), key=lambda i: p_values[i])
        corrected = [1.0] * n

        # BH procedure
        for rank, idx in enumerate(sorted_indices):
            corrected[idx] = min(p_values[idx] * n / (rank + 1), 1.0)

        rejected = [p < alpha for p in corrected]

        # Estimate FDR
        if sum(rejected) > 0:
            expected_false_positives = alpha * n
            fdr = expected_false_positives / sum(rejected)
        else:
            fdr = 0.0

        return CorrectionResult(
            original_p_values=p_values,
            corrected_p_values=corrected,
            rejected=rejected,
            method=CorrectionMethod.BENJAMINI_HOCHBERG,
            alpha=alpha,
            num_rejections=sum(rejected),
            num_tests=n,
            false_discovery_rate=min(fdr, 1.0),
        )

    @staticmethod
    def benjamini_yekutieli(
        p_values: List[float],
        alpha: float = 0.05,
    ) -> CorrectionResult:
        """
        Benjamini-Yekutieli FDR correction.

        More conservative than BH, works under dependency.

        Args:
            p_values: List of p-values
            alpha: Significance level

        Returns:
            CorrectionResult
        """
        n = len(p_values)

        # Harmonic sum
        harmonic_sum = sum(1 / (i + 1) for i in range(n))

        # Sort p-values with indices
        sorted_indices = sorted(range(n), key=lambda i: p_values[i])
        corrected = [1.0] * n

        # BY procedure
        for rank, idx in enumerate(sorted_indices):
            corrected[idx] = min(
                p_values[idx] * n / ((rank + 1) * harmonic_sum),
                1.0
            )

        rejected = [p < alpha for p in corrected]

        return CorrectionResult(
            original_p_values=p_values,
            corrected_p_values=corrected,
            rejected=rejected,
            method=CorrectionMethod.BENJAMINI_YEKUTIELI,
            alpha=alpha,
            num_rejections=sum(rejected),
            num_tests=n,
        )

    @staticmethod
    def storey_q_value(
        p_values: List[float],
        alpha: float = 0.05,
        pi0: Optional[float] = None,
    ) -> CorrectionResult:
        """
        Storey's q-value method.

        Estimates the proportion of true null hypotheses (pi0)
        for more accurate FDR control.

        Args:
            p_values: List of p-values
            alpha: Significance level
            pi0: Estimate of true null proportion (None = auto-estimate)

        Returns:
            CorrectionResult
        """
        n = len(p_values)
        p_values = np.array(p_values)

        # Estimate pi0 if not provided
        if pi0 is None:
            # Bootstrap method
            lambda_range = np.arange(0.05, 0.96, 0.01)
            pi0_estimates = []
            for lam in lambda_range:
                W = np.sum(p_values > lam)
                pi0_estimates.append(W / (n * (1 - lam)))

            # Use minimum estimate
            pi0 = min(pi0_estimates) if pi0_estimates else 1.0
            pi0 = min(pi0, 1.0)

        # Sort p-values
        sorted_indices = np.argsort(p_values)
        corrected = np.ones(n)

        # Calculate q-values
        for rank, idx in enumerate(sorted_indices):
            q = (pi0 * n * p_values[idx]) / (rank + 1)
            corrected[idx] = min(q, 1.0)

        # Monotonicity constraint
        for i in range(1, n):
            corrected[sorted_indices[i]] = min(
                corrected[sorted_indices[i]],
                corrected[sorted_indices[i - 1]]
            )

        rejected = [p < alpha for p in corrected]

        return CorrectionResult(
            original_p_values=p_values.tolist(),
            corrected_p_values=corrected.tolist(),
            rejected=rejected,
            method=CorrectionMethod.STOREY,
            alpha=alpha,
            num_rejections=sum(rejected),
            num_tests=n,
            false_discovery_rate=pi0,
        )

    @classmethod
    def apply_correction(
        cls,
        p_values: List[float],
        method: CorrectionMethod = CorrectionMethod.BONFERRONI,
        alpha: float = 0.05,
        **kwargs,
    ) -> CorrectionResult:
        """
        Apply specified correction method.

        Args:
            p_values: List of p-values
            method: Correction method to use
            alpha: Significance level
            **kwargs: Additional parameters for specific methods

        Returns:
            CorrectionResult
        """
        if method == CorrectionMethod.NONE:
            return CorrectionResult(
                original_p_values=p_values,
                corrected_p_values=p_values,
                rejected=[p < alpha for p in p_values],
                method=method,
                alpha=alpha,
                num_rejections=sum(p < alpha for p in p_values),
                num_tests=len(p_values),
            )

        elif method == CorrectionMethod.BONFERRONI:
            return cls.bonferroni(p_values, alpha)

        elif method == CorrectionMethod.HOLM:
            return cls.holm(p_values, alpha)

        elif method == CorrectionMethod.HOCHBERG:
            return cls.hochberg(p_values, alpha)

        elif method == CorrectionMethod.SIDAK:
            return cls.sidak(p_values, alpha)

        elif method == CorrectionMethod.BENJAMINI_HOCHBERG:
            return cls.benjamini_hochberg(p_values, alpha)

        elif method == CorrectionMethod.BENJAMINI_YEKUTIELI:
            return cls.benjamini_yekutieli(p_values, alpha)

        elif method == CorrectionMethod.STOREY:
            return cls.storey_q_value(p_values, alpha, **kwargs)

        else:
            raise ValueError(f"Unknown correction method: {method}")

    @staticmethod
    def compare_methods(
        p_values: List[float],
        alpha: float = 0.05,
    ) -> Dict[CorrectionMethod, CorrectionResult]:
        """
        Compare all correction methods on the same p-values.

        Args:
            p_values: List of p-values
            alpha: Significance level

        Returns:
            Dictionary of method to CorrectionResult
        """
        results = {}

        methods = [
            CorrectionMethod.NONE,
            CorrectionMethod.BONFERRONI,
            CorrectionMethod.HOLM,
            CorrectionMethod.HOCHBERG,
            CorrectionMethod.SIDAK,
            CorrectionMethod.BENJAMINI_HOCHBERG,
            CorrectionMethod.BENJAMINI_YEKUTIELI,
        ]

        for method in methods:
            try:
                result = MultipleComparisonCorrection.apply_correction(
                    p_values, method, alpha
                )
                results[method] = result
            except Exception as e:
                logger.warning(f"Error applying {method}: {e}")

        return results


# ============================================================================
# Convenience Functions
# ============================================================================

def correct_p_values(
    p_values: List[float],
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> List[float]:
    """
    Simple function to correct p-values.

    Args:
        p_values: List of p-values
        method: Correction method name
        alpha: Significance level

    Returns:
        List of corrected p-values
    """
    correction_method = CorrectionMethod(method)
    result = MultipleComparisonCorrection.apply_correction(
        p_values, correction_method, alpha
    )
    return result.corrected_p_values


# Export
__all__ = [
    # Enums
    "CorrectionMethod",
    # Data classes
    "CorrectionResult",
    # Main class
    "MultipleComparisonCorrection",
    # Functions
    "correct_p_values",
]
